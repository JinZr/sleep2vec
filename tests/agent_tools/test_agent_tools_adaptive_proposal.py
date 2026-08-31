from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import pytest
import yaml

from agent_tools import adaptive_hparam, adaptive_proposals, hparam_runtime, manifests, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _adaptive_recipe,
    _agent_recipe,
    _mark_round_terminal,
    _read_table,
    _run,
    _test_selected_adaptive_recipe,
    _write_agent_configuration_submission,
    _write_agent_submission,
    _write_checkpoint_test_manifest,
    _write_fake_manifest,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def test_adaptive_projection_publication_is_atomic_and_retryable(tmp_path: Path, monkeypatch):
    target = tmp_path / "adaptive" / "projection.yaml"
    content = b"decision: accepted\n"
    original_rename = adaptive_hparam.exp_io._rename_noreplace_at
    monkeypatch.setattr(
        adaptive_hparam.exp_io,
        "_rename_noreplace_at",
        lambda *_args: (_ for _ in ()).throw(OSError("publication interrupted")),
    )

    with pytest.raises(OSError, match="publication interrupted"):
        adaptive_hparam._write_exact_bytes(target, content, managed_root=tmp_path)

    assert not target.exists()
    monkeypatch.setattr(adaptive_hparam.exp_io, "_rename_noreplace_at", original_rename)
    adaptive_hparam._write_exact_bytes(target, content, managed_root=tmp_path)
    adaptive_hparam._write_exact_bytes(target, content, managed_root=tmp_path)
    assert target.read_bytes() == content

    outside = tmp_path / "outside.yaml"
    outside.write_bytes(b"original\n")
    alias = tmp_path / "adaptive" / "alias.yaml"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="Existing adaptive projection differs"):
        adaptive_hparam._write_exact_bytes(alias, content, managed_root=tmp_path)
    assert outside.read_bytes() == b"original\n"


@pytest.mark.parametrize("explicit_strategy", [False, True])
def test_agent_proposal_waits_for_terminal_round_then_writes_deterministic_snapshot(
    tmp_path: Path, explicit_strategy: bool
):
    recipe = _agent_recipe(tmp_path, explicit_strategy=explicit_strategy)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    step_manifests = list((tmp_path / "steps").glob("*/step.yaml"))
    assert len(step_manifests) == 1
    assert yaml.safe_load(step_manifests[0].read_text())["plan_controller"] == "adaptive"

    assert adaptive_hparam.adaptive_step(workflow_dir) is None
    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()

    _write_fake_manifest(workflow_dir, score=0.73)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    first_bytes = input_path.read_bytes()
    proposal_input = json.loads(first_bytes)
    assert proposal_input["input"]["source_round"] == 0
    assert proposal_input["input"]["target_round"] == 1
    assert proposal_input["input"]["parameter_envelopes"]["runtime.lr"] == {
        "kind": "number",
        "min": 5e-7,
        "max": 2e-6,
    }
    effective_recipe = adaptive_hparam.load_recipe_with_base(recipe)
    config_path = adaptive_hparam.resolve_repo_path(effective_recipe["inputs"]["config"])
    assert config_path is not None
    assert proposal_input["schema_version"] == 2
    assert proposal_input["input"]["source_config_sha256"] == adaptive_hparam.file_sha256(config_path)
    assert Path(proposal_input["expected_proposal_path"]).name == (
        f"round_001--{proposal_input['request_id'][7:19]}.json"
    )

    request_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]
    assert len(request_events) == 1
    assert request_events[0]["input_sha256"] == adaptive_hparam.file_sha256(input_path)

    assert adaptive_hparam.adaptive_step(workflow_dir) == input_path
    assert input_path.read_bytes() == first_bytes
    assert (tmp_path / "events.jsonl").read_text().count('"event_type": "agent_proposal_requested"') == 1


def test_agent_proposal_request_recovers_missing_issuance_for_existing_snapshot(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    first_bytes = input_path.read_bytes()
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events = [event for event in events if event.get("event_type") != "agent_proposal_requested"]
    events_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))

    # Simulate a crash after the immutable input write but before the matching issuance append.
    assert adaptive_hparam.adaptive_step(workflow_dir) == input_path

    assert input_path.read_bytes() == first_bytes
    request_events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]
    assert len(request_events) == 1
    assert request_events[0]["input_sha256"] == adaptive_hparam.file_sha256(input_path)


def test_agent_proposal_request_uses_blocking_events_lock(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    real_lock = adaptive_hparam.exp_io.blocking_file_lock
    locked_paths = []

    def tracked_lock(path):
        locked_paths.append(Path(path))
        return real_lock(path)

    monkeypatch.setattr(adaptive_hparam.exp_io, "blocking_file_lock", tracked_lock)

    assert adaptive_hparam.adaptive_step(workflow_dir) is not None
    assert locked_paths.count(tmp_path / "events.jsonl.lock") == 1


def test_agent_proposal_request_treats_one_exact_issuance_as_idempotent(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    events_path = tmp_path / "events.jsonl"
    request_events_before = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]

    assert adaptive_hparam.adaptive_step(workflow_dir) == input_path

    request_events_after = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]
    assert request_events_after == request_events_before


@pytest.mark.parametrize("shared_binding", ["input_path", "proposal_path", "target_round"])
def test_agent_proposal_request_rejects_wrong_request_id_sharing_a_binding(tmp_path: Path, shared_binding: str):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    request_index = next(
        index for index, event in enumerate(events) if event.get("event_type") == "agent_proposal_requested"
    )
    conflicting = dict(events[request_index])
    conflicting["request_id"] = "sha256:" + "f" * 64
    if shared_binding != "input_path":
        conflicting["input_path"] = str(input_path.with_name("other-input.json"))
    if shared_binding != "proposal_path":
        conflicting["proposal_path"] = str(Path(conflicting["proposal_path"]).with_name("other-proposal.json"))
    if shared_binding != "target_round":
        conflicting["target_round"] = 2
    events[request_index] = conflicting
    events_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))

    with pytest.raises(ValueError, match="differs from its phase-one issuance"):
        adaptive_hparam.adaptive_step(workflow_dir)

    request_events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]
    assert request_events == [conflicting]


def test_agent_proposal_request_allows_same_target_round_in_another_workflow(tmp_path: Path):
    first_recipe = _agent_recipe(tmp_path)
    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["name"] = "unit_adaptive_second"
    second_payload["step"]["id"] = "unit-hparam-tune-second"
    second_recipe = tmp_path / "adaptive_tune_second.yaml"
    second_recipe.write_text(yaml.safe_dump(second_payload))
    first_workflow = tmp_path / "workflow-first"
    second_workflow = tmp_path / "workflow-second"

    for recipe, workflow in ((first_recipe, first_workflow), (second_recipe, second_workflow)):
        result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))
        assert result.returncode == 0, result.stderr
        _write_fake_manifest(workflow)
        _mark_round_terminal(workflow, tmp_path)

    first_input = adaptive_hparam.adaptive_step(first_workflow)
    second_input = adaptive_hparam.adaptive_step(second_workflow)

    assert first_input is not None
    assert second_input is not None
    assert json.loads(first_input.read_text())["input"]["target_round"] == 1
    assert json.loads(second_input.read_text())["input"]["target_round"] == 1
    request_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    ]
    assert {event["input_path"] for event in request_events} == {str(first_input), str(second_input)}


def test_agent_proposal_request_rejects_duplicate_exact_issuance(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    events_path = tmp_path / "events.jsonl"
    request_event = next(
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "agent_proposal_requested"
    )
    with events_path.open("a") as file_obj:
        file_obj.write(json.dumps(request_event, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="no unique phase-one issuance"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert events_path.read_text().count('"event_type": "agent_proposal_requested"') == 2


def test_agent_proposal_request_validates_events_lock_path(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    (tmp_path / "events.jsonl.lock").symlink_to(tmp_path / "outside-lock")

    with pytest.raises(ValueError, match="independent regular files"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()


def test_agent_proposal_request_rejects_conflicting_existing_issuance(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    request_event = next(event for event in events if event.get("event_type") == "agent_proposal_requested")
    request_event["input_sha256"] = "0" * 64
    events_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))

    with pytest.raises(ValueError, match="differs from its phase-one issuance"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert events_path.read_text().count('"event_type": "agent_proposal_requested"') == 1


def test_agent_proposal_can_request_after_all_runs_fail_without_a_score(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    run = json.loads((workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}],
    )

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    row = json.loads(input_path.read_text())["input"]["digest_rows"][0]
    assert row["run_id"] == run["run_id"]
    assert row["status"] == "failed"
    assert "test_auroc" not in row


def test_agent_proposal_authoritative_snapshot_normalizes_sparse_digest_fields(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    runs = json.loads((workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": runs[1]["step_id"], "run_id": runs[1]["run_id"], "status": "failed"}],
    )

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    digest_rows = json.loads(input_path.read_text())["input"]["digest_rows"]
    assert digest_rows[0]["best_model_score"] == "0.5"
    assert digest_rows[1]["best_model_score"] == ""
    proposal_path = _write_agent_submission(input_path)
    assert adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path) == proposal_path


def test_agent_proposal_preview_is_read_only_and_execute_uses_bound_snapshot(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir, score=0.73)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    assert adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path) == proposal_path
    assert events_path.read_bytes() == events_before
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()

    (workflow_dir / "adaptive" / "digests" / "round_000.csv").write_text("run_id,status\nforeign,finished\n")
    monkeypatch.setattr(
        adaptive_hparam,
        "digest_hparam_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("phase 2 must not refresh digest")),
    )
    monkeypatch.setattr(
        adaptive_hparam,
        "_latest_digest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("phase 2 must not read latest digest")),
    )

    def fake_launch(run_dir, *, dry_run=True):
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    suggestion = adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    suggestion_payload = yaml.safe_load(suggestion.read_text())
    accepted = json.loads((workflow_dir / "adaptive" / "proposals" / "round_001.json").read_text())
    assert suggestion_payload["search"]["parameters"]["runtime.lr"] == [5e-7]
    assert suggestion_payload["search"]["max_runs"] == 1
    assert accepted["request_id"] == json.loads(input_path.read_text())["request_id"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    assert "agent_proposal_accepted" in events_path.read_text()
    with pytest.raises(ValueError, match="round binding is stale"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path)


def test_agent_proposal_recovers_exact_published_unregistered_round(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    commit_plan = adaptive_hparam.plan_hparam.commit_hparam_plan
    commit_calls = 0

    def interrupt_first_commit(*args, **kwargs):
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            raise RuntimeError("registration interrupted")
        return commit_plan(*args, **kwargs)

    def fake_launch(run_dir, *, dry_run=True):
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in runs],
        )
        return Path(run_dir) / "launch_manifest.tsv"

    monkeypatch.setattr(adaptive_hparam.plan_hparam, "commit_hparam_plan", interrupt_first_commit)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    with pytest.raises(RuntimeError, match="registration interrupted"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    accepted_path = workflow_dir / "adaptive" / "proposals" / "round_001.json"
    suggestion_path = workflow_dir / "adaptive" / "suggestions" / "round_001.yaml"
    frozen_bytes = {
        "plan": (next_dir / "plan.json").read_bytes(),
        "accepted": accepted_path.read_bytes(),
        "suggestion": suggestion_path.read_bytes(),
    }
    assert [row["round"] for row in _read_table(workflow_dir / "adaptive" / "run_registry.tsv")] == ["0"]

    adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    assert (next_dir / "plan.json").read_bytes() == frozen_bytes["plan"]
    assert accepted_path.read_bytes() == frozen_bytes["accepted"]
    assert suggestion_path.read_bytes() == frozen_bytes["suggestion"]
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    assert [row["round"] for row in _read_table(workflow_dir / "adaptive" / "run_registry.tsv")] == ["0", "1"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("agent_proposal_accepted") == 1


def test_concurrent_agent_proposal_execute_serializes_round_projections(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    stage_round = adaptive_hparam._stage_round
    first_staging = threading.Event()
    second_staging = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    stage_calls = 0

    def pause_first_stage(*args, **kwargs):
        nonlocal stage_calls
        with state_lock:
            stage_calls += 1
            call = stage_calls
        if call == 1:
            first_staging.set()
            assert release_first.wait(timeout=10)
        else:
            second_staging.set()
        return stage_round(*args, **kwargs)

    def fake_launch(run_dir, *, dry_run=True):
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in runs],
        )
        return Path(run_dir) / "launch_manifest.tsv"

    monkeypatch.setattr(adaptive_hparam, "_stage_round", pause_first_stage)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    results = []
    errors = []

    def execute():
        try:
            results.append(adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=execute)
    second = threading.Thread(target=execute)
    first.start()
    assert first_staging.wait(timeout=10)
    second.start()
    try:
        assert not second_staging.wait(timeout=0.5)
    finally:
        release_first.set()
    first.join(timeout=30)
    second.join(timeout=30)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 1
    assert len(errors) == 1
    assert "round binding is stale" in str(errors[0])
    assert stage_calls == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("agent_proposal_accepted") == 1


def test_agent_proposal_rejects_tampered_snapshot_before_acceptance(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    proposal_input = json.loads(input_path.read_text())
    proposal_input["input"]["remaining_budget"]["runs"] -= 1
    input_path.write_text(json.dumps(proposal_input, indent=2, sort_keys=True) + "\n")
    events_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="request_id does not match"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    assert (tmp_path / "events.jsonl").read_bytes() == events_before
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


def test_agent_proposal_rejects_forged_self_consistent_snapshot_and_issuance(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    real_input = adaptive_hparam.adaptive_step(workflow_dir)
    assert real_input is not None
    forged = json.loads(real_input.read_text())
    forged["input"]["remaining_budget"]["round_size"] = 2
    forged["input"]["parameter_envelopes"]["runtime.lr"]["max"] = 1.0
    forged["request_id"] = adaptive_hparam.adaptive_proposals.proposal_request_id(forged["input"])
    id12 = forged["request_id"][7:19]
    forged_input = workflow_dir / "adaptive" / "proposal_inputs" / f"round_001--{id12}.json"
    forged_proposal = workflow_dir / "adaptive" / "proposal_submissions" / f"round_001--{id12}.json"
    forged["expected_proposal_path"] = str(forged_proposal)
    forged_input.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
    _write_agent_submission(forged_input, lr=[0.5])

    with pytest.raises(ValueError, match="issuance record"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=forged_proposal)

    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events = [event for event in events if event.get("event_type") != "agent_proposal_requested"]
    events.append(
        {
            "event_type": "agent_proposal_requested",
            "source_round": 0,
            "target_round": 1,
            "request_id": forged["request_id"],
            "input_path": str(forged_input),
            "input_sha256": adaptive_hparam.file_sha256(forged_input),
            "proposal_path": str(forged_proposal),
        }
    )
    events_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))

    with pytest.raises(ValueError, match="current authoritative snapshot"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=forged_proposal)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


@pytest.mark.parametrize("execute", [False, True])
def test_agent_proposal_rejects_source_config_drift_after_snapshot(tmp_path: Path, execute: bool):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    effective_recipe = adaptive_hparam.load_recipe_with_base(recipe)
    config_path = adaptive_hparam.resolve_repo_path(effective_recipe["inputs"]["config"])
    assert config_path is not None
    config_path.write_text(config_path.read_text() + "\n# changed after proposal snapshot\n")

    with pytest.raises(ValueError, match="source config changed"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=execute)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


def test_agent_proposal_rechecks_source_config_after_candidate_preflight(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    effective_recipe = adaptive_hparam.load_recipe_with_base(recipe)
    config_path = adaptive_hparam.resolve_repo_path(effective_recipe["inputs"]["config"])
    assert config_path is not None
    real_preflight = adaptive_hparam.preflight_plan

    def mutate_after_candidate_preflight(*args, **kwargs):
        result = real_preflight(*args, **kwargs)
        recipe_path = Path(kwargs.get("recipe_path") or args[0])
        if recipe_path.name == "suggested.yaml":
            config_path.write_text(config_path.read_text() + "\n# changed during candidate preflight\n")
        return result

    monkeypatch.setattr(adaptive_hparam, "preflight_plan", mutate_after_candidate_preflight)
    events_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="source config changed"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path)

    assert (tmp_path / "events.jsonl").read_bytes() == events_before
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


@pytest.mark.parametrize("drift", ["source_recipe", "workflow_recipe_path"])
def test_agent_proposal_refreshes_source_contract_after_candidate_preflight(tmp_path: Path, monkeypatch, drift: str):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path, lr=[2e-6])
    real_preflight = adaptive_hparam.preflight_plan

    def narrow_bounds_after_candidate_preflight(*args, **kwargs):
        result = real_preflight(*args, **kwargs)
        recipe_path = Path(kwargs.get("recipe_path") or args[0])
        if recipe_path.name == "suggested.yaml":
            payload = yaml.safe_load(recipe.read_text())
            payload["adaptive"]["suggest"]["bounds"]["runtime.lr"] = [5e-7, 1e-6]
            if drift == "source_recipe":
                recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
            else:
                replacement = tmp_path / "replacement-adaptive.yaml"
                replacement.write_text(yaml.safe_dump(payload, sort_keys=False))
                workflow_path = workflow_dir / "adaptive" / "workflow.json"
                workflow = json.loads(workflow_path.read_text())
                workflow["recipe_path"] = str(replacement)
                manifests.write_json(workflow_path, workflow)
        return result

    launches = []
    monkeypatch.setattr(adaptive_hparam, "preflight_plan", narrow_bounds_after_candidate_preflight)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", lambda *_args, **_kwargs: launches.append(True))

    with pytest.raises(ValueError, match="source recipe changed"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    assert launches == []
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


def test_agent_proposal_rebuilds_candidate_from_refreshed_base_and_local_pair(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    real_preflight = adaptive_hparam.preflight_plan
    mutated = False

    def offset_base_and_local_after_candidate_preflight(*args, **kwargs):
        nonlocal mutated
        result = real_preflight(*args, **kwargs)
        recipe_path = Path(kwargs.get("recipe_path") or args[0])
        if recipe_path.name == "suggested.yaml" and not mutated:
            mutated = True
            local_payload = yaml.safe_load(recipe.read_text())
            base_path = Path(local_payload["base_recipe"])
            base_payload = yaml.safe_load(base_path.read_text())
            base_payload["runtime"]["devices"] = [7]
            base_path.write_text(yaml.safe_dump(base_payload, sort_keys=False))
            # This local override keeps the effective snapshot at devices=[0].
            local_payload["runtime"] = {"devices": [0]}
            recipe.write_text(yaml.safe_dump(local_payload, sort_keys=False))
        return result

    def fake_launch(run_dir, *, dry_run=True):
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "preflight_plan", offset_base_and_local_after_candidate_preflight)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    plan = json.loads((workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())
    assert plan["recipe"]["runtime"]["devices"] == [0]
    assert plan["recipe"]["_local_recipe"]["runtime"]["devices"] == [0]
    assert plan["recipe"]["_base_recipe"]["runtime"]["devices"] == [7]


def test_agent_proposal_materializes_bound_recipe_and_config_bytes(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    effective_recipe = adaptive_hparam.load_recipe_with_base(recipe)
    config_path = adaptive_hparam.resolve_repo_path(effective_recipe["inputs"]["config"])
    assert config_path is not None
    original_max_tokens = yaml.safe_load(config_path.read_text())["data"]["max_tokens"]
    real_write_round_recipe = adaptive_hparam._write_round_recipe

    def mutate_source_before_round_recipe(*args, **kwargs):
        source_config = yaml.safe_load(config_path.read_text())
        source_config["data"]["max_tokens"] = 999
        config_path.write_text(yaml.safe_dump(source_config))
        return real_write_round_recipe(*args, **kwargs)

    monkeypatch.setattr(adaptive_hparam, "_write_round_recipe", mutate_source_before_round_recipe)

    def fake_launch(run_dir, *, dry_run=True):
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    plan = json.loads((next_dir / "plan.json").read_text())
    run = plan["runs"][0]
    assert run["runtime.lr"] == 5e-7
    assert yaml.safe_load(Path(run["config"]).read_text())["data"]["max_tokens"] == original_max_tokens
    round_recipe = yaml.safe_load((next_dir / "round_recipe.yaml").read_text())
    assert round_recipe["inputs"]["config"] == str(next_dir / "source_config.yaml")
    assert yaml.safe_load(config_path.read_text())["data"]["max_tokens"] == 999
    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert suggestion["search"]["parameters"]["runtime.lr"] == [5e-7]


def test_agent_proposal_rejects_frozen_config_replacement_inside_plan_builder(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    real_build_plan = adaptive_hparam.build_plan

    def replace_frozen_config_before_builder_read(**kwargs):
        bound_config = Path(kwargs["output_dir"]) / "source_config.yaml"
        config = yaml.safe_load(bound_config.read_text())
        config["data"]["max_tokens"] = 999
        bound_config.write_text(yaml.safe_dump(config))
        return real_build_plan(**kwargs)

    launches = []
    monkeypatch.setattr(adaptive_hparam, "build_plan", replace_frozen_config_before_builder_read)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", lambda *_args, **_kwargs: launches.append(True))

    with pytest.raises(ValueError, match="bound SHA-256"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    assert not (next_dir / "plan.json").exists()
    assert {row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")} == {"run-000"}
    assert launches == []


def test_agent_proposal_recipe_binding_is_json_type_strict(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    recipe_payload["search"]["parameters"]["yaml:/model/projection/enabled"] = [True]
    recipe.write_text(yaml.safe_dump(recipe_payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    proposal = json.loads(proposal_path.read_text())
    proposal["parameters"]["yaml:/model/projection/enabled"] = [True]
    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")
    real_build_plan = adaptive_hparam.build_plan

    def replace_boolean_with_integer(**kwargs):
        round_recipe = Path(kwargs["recipe_path"])
        payload = yaml.safe_load(round_recipe.read_text())
        payload["search"]["parameters"]["yaml:/model/projection/enabled"] = [1]
        round_recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
        return real_build_plan(**kwargs)

    launches = []
    monkeypatch.setattr(adaptive_hparam, "build_plan", replace_boolean_with_integer)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", lambda *_args, **_kwargs: launches.append(True))

    with pytest.raises(ValueError, match="bound adaptive recipe"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    assert not (next_dir / "plan.json").exists()
    assert {row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")} == {"run-000"}
    assert launches == []


@pytest.mark.parametrize("recipe_layer", ["round", "base"])
def test_agent_proposal_rejects_recipe_replacement_inside_plan_builder(tmp_path: Path, monkeypatch, recipe_layer: str):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    real_build_plan = adaptive_hparam.build_plan

    def replace_recipe_before_builder_read(**kwargs):
        round_recipe = Path(kwargs["recipe_path"])
        payload = yaml.safe_load(round_recipe.read_text())
        if recipe_layer == "round":
            payload["search"]["parameters"]["runtime.lr"] = [0.5]
            round_recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
        else:
            base_recipe = Path(payload["base_recipe"])
            base_payload = yaml.safe_load(base_recipe.read_text())
            base_payload["inputs"]["label_name"] = "forged"
            base_recipe.write_text(yaml.safe_dump(base_payload, sort_keys=False))
        return real_build_plan(**kwargs)

    launches = []
    monkeypatch.setattr(adaptive_hparam, "build_plan", replace_recipe_before_builder_read)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", lambda *_args, **_kwargs: launches.append(True))

    with pytest.raises(ValueError, match="bound adaptive recipe"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    assert not (next_dir / "plan.json").exists()
    assert {row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")} == {"run-000"}
    assert launches == []


def test_agent_proposal_rejects_tampered_expected_path_even_when_request_id_is_unchanged(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    forged_path = proposal_path.with_name("forged.json")
    forged_path.write_bytes(proposal_path.read_bytes())
    proposal_input = json.loads(input_path.read_text())
    proposal_input["expected_proposal_path"] = str(forged_path)
    input_path.write_text(json.dumps(proposal_input, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="expected path does not match"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=forged_path)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


def test_agent_proposal_rejects_source_recipe_drift_after_snapshot(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["name"] = "changed_after_snapshot"
    recipe.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="source recipe changed"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()


def test_agent_proposal_rechecks_live_budget_after_snapshot(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    monkeypatch.setattr(adaptive_hparam, "_budget_exhausted", lambda *_args, **_kwargs: True)

    with pytest.raises(ValueError, match="no longer fits"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()


def test_agent_proposal_zero_start_recovery_uses_a_fresh_target_round(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    first_input = adaptive_hparam.adaptive_step(workflow_dir)
    assert first_input is not None
    first_proposal = _write_agent_submission(first_input)
    launch_statuses = iter(["pending", "launched"])
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: next(launch_statuses))

    with pytest.raises(RuntimeError, match=r"started no runs.*was not committed"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=first_proposal, execute=True)

    first_attempt = workflow_dir / "adaptive" / "rounds" / "round_001"
    assert first_attempt.exists()
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    second_input = adaptive_hparam.adaptive_step(workflow_dir)
    assert second_input is not None
    second_snapshot = json.loads(second_input.read_text())
    assert second_snapshot["input"]["source_round"] == 0
    assert second_snapshot["input"]["target_round"] == 2
    second_proposal = _write_agent_submission(second_input)
    blocked_rationale = workflow_dir / "adaptive" / "suggestions" / "round_002.md"
    blocked_rationale.symlink_to(tmp_path / "missing-rationale.md")
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match="independent regular files"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=second_proposal, execute=True)

    assert events_path.read_bytes() == events_before
    assert (
        next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")["status"]
        == "pending"
    )
    assert not (workflow_dir / "adaptive" / "proposals" / "round_002.json").exists()
    blocked_rationale.unlink()

    adaptive_hparam.adaptive_step(workflow_dir, proposal_path=second_proposal, execute=True)

    assert adaptive_hparam._latest_round_index(workflow_dir) == 2
    assert (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "finished", "run-001": "superseded", "run-002": "launched"}


@pytest.mark.parametrize("protocol_file", ["input", "proposal"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_agent_proposal_rejects_aliased_protocol_file(tmp_path: Path, protocol_file: str, alias_kind: str):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    protocol_path = input_path if protocol_file == "input" else proposal_path
    backing = tmp_path / f"proposal-{protocol_file}-backing.json"
    protocol_path.rename(backing)
    if alias_kind == "symlink":
        protocol_path.symlink_to(backing)
    else:
        os.link(backing, protocol_path)

    with pytest.raises(ValueError, match="independent regular files"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()


def test_agent_proposal_rejects_submission_changed_during_validation(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    file_sha256 = adaptive_hparam.file_sha256

    def mutate_before_recheck(path):
        if Path(path) == proposal_path:
            proposal_path.write_text("{}\n")
        return file_sha256(path)

    monkeypatch.setattr(adaptive_hparam, "file_sha256", mutate_before_recheck)

    with pytest.raises(ValueError, match="changed during validation"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()


def test_hparam_suggest_rejects_agent_strategy_without_reusing_latest_digest(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(
        adaptive_hparam,
        "_latest_digest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("latest digest must not be read")),
    )

    with pytest.raises(ValueError, match="generated by hparam-adaptive-step"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()


def test_agent_proposal_cli_reports_waiting_and_side_effect_free_preview(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr

    waiting = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))
    assert waiting.returncode == 0, waiting.stderr
    assert waiting.stdout.strip() == "waiting_for_round_terminal"

    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    requested = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))
    assert requested.returncode == 0, requested.stderr
    input_path = Path(requested.stdout.strip().removeprefix("Wrote "))
    proposal_path = _write_agent_submission(input_path)
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    preview = _run(
        "hparam-adaptive-step",
        "--workflow-dir",
        str(workflow_dir),
        "--proposal",
        str(proposal_path),
    )

    assert preview.returncode == 0, preview.stderr
    assert preview.stdout.strip() == f"Validated {proposal_path}"
    assert events_path.read_bytes() == events_before
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()


def test_agent_proposal_execute_requires_submission_before_digest(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(
        adaptive_hparam,
        "digest_hparam_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("digest must not run")),
    )

    with pytest.raises(ValueError, match="requires --proposal"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)


@pytest.mark.parametrize("command", ["step", "suggest", "loop"])
def test_adaptive_commands_reject_disabled_source_before_writing(tmp_path: Path, command: str):
    recipe = _agent_recipe(tmp_path, explicit_strategy=False)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["enabled"] = False
    recipe.write_text(yaml.safe_dump(payload))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    run = {
        "step": lambda: adaptive_hparam.adaptive_step(workflow_dir),
        "suggest": lambda: adaptive_hparam.suggest_next_round(workflow_dir),
        "loop": lambda: adaptive_hparam.adaptive_loop(workflow_dir),
    }[command]
    with pytest.raises(ValueError, match="adaptive.enabled must be true"):
        run()

    assert events_path.read_bytes() == events_before
    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()
    assert not (workflow_dir / "adaptive" / "suggestions").exists()


def test_agent_proposal_loop_fails_without_writing_an_event(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    with pytest.raises(ValueError, match="does not support agent_proposal"):
        adaptive_hparam.adaptive_loop(workflow_dir)

    assert events_path.read_bytes() == before


def test_agent_proposal_configuration_points_execute_as_exact_runs(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["round_size"] = 2  # two-point round must fit round_size
    recipe.write_text(yaml.safe_dump(payload))
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr
    _write_fake_manifest(workflow_dir, score=0.73)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_configuration_submission(input_path)
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    assert adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path) == proposal_path
    assert events_path.read_bytes() == events_before  # preview is side-effect free

    def fake_launch(run_dir, *, dry_run=True):
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    suggestion = adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=True)

    suggestion_payload = yaml.safe_load(suggestion.read_text())
    assert suggestion_payload["search"]["configurations"] == [
        {"runtime.lr": 5e-7, "yaml:/model/head/name": "classification"},
        {"runtime.lr": 2e-6, "yaml:/model/head/name": "classification"},
    ]
    assert suggestion_payload["search"]["max_runs"] == 2
    assert "parameters" not in suggestion_payload["search"]

    plan_runs = json.loads((workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())["runs"]
    assert len(plan_runs) == 2  # two points, not the 2x1 product per key
    assert [run["runtime.lr"] for run in plan_runs] == [5e-7, 2e-6]

    accepted = json.loads((workflow_dir / "adaptive" / "proposals" / "round_001.json").read_text())
    assert accepted["configurations"] == suggestion_payload["search"]["configurations"]
    assert "parameters" not in accepted
    rationale = (workflow_dir / "adaptive" / "suggestions" / "round_001.md").read_text()
    assert "## Configurations" in rationale
    assert "point 0" in rationale and "point 1" in rationale
    assert "agent_proposal_accepted" in events_path.read_text()


@pytest.mark.parametrize("execute", [False, True])
def test_agent_proposal_rejects_envelope_valid_joint_config_before_acceptance(
    tmp_path: Path, monkeypatch, execute: bool
):
    recipe = _agent_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"].update(
        {
            "yaml:/model/cls/downstream": ["tokens", "cls"],
            "yaml:/model/cls/embedding_type": ["bert", "none"],
        }
    )
    assert payload["search"]["max_runs"] == 1
    recipe.write_text(yaml.safe_dump(payload))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    _write_fake_manifest(workflow_dir)
    _mark_round_terminal(workflow_dir, tmp_path)
    input_path = adaptive_hparam.adaptive_step(workflow_dir)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    proposal = json.loads(proposal_path.read_text())
    proposal["parameters"].update({"yaml:/model/cls/downstream": ["cls"], "yaml:/model/cls/embedding_type": ["none"]})
    proposal_path.write_text(json.dumps(proposal))
    normalized = adaptive_proposals.validate_proposal(proposal, json.loads(input_path.read_text()))
    assert normalized["max_runs"] == 1
    events_before = (tmp_path / "events.jsonl").read_bytes()
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        adaptive_hparam,
        "launch_hparam_runs",
        lambda *_args, **_kwargs: pytest.fail("Invalid agent candidate reached launch"),
    )

    with pytest.raises(RuntimeError, match="Agent proposal failed preflight.*model.cls.embedding_type must be set"):
        adaptive_hparam.adaptive_step(workflow_dir, proposal_path=proposal_path, execute=execute)

    assert (tmp_path / "events.jsonl").read_bytes() == events_before
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert not (workflow_dir / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


def test_explicit_best_neighborhood_uses_existing_numeric_neighbors(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)

    digest = _run("hparam-digest", "--run-dir", str(workflow_dir))
    suggest = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert digest.returncode == 0, digest.stderr
    assert suggest.returncode == 0, suggest.stderr
    rows = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")
    assert rows[0]["test_auroc"] == "0.73"
    assert rows[0]["checkpoint_path"].endswith("epoch=3.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]
    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert suggestion["search"]["parameters"]["runtime.lr"] == [5e-07, 1e-06, 1.5e-06]
    assert "external_optimized: true" in (workflow_dir / "adaptive" / "digests" / "round_000.md").read_text()
    incumbents = _read_table(workflow_dir / "adaptive" / "incumbents.tsv")
    assert incumbents[-1]["objective_score"] == "0.73"


@pytest.mark.parametrize(
    ("objective_mode", "scores", "top_level_score", "expected_score"),
    [
        ("max", {1: 0.5, 2: 0.9, 3: 0.9}, 0.99, "0.9"),
        ("min", {1: 0.5, 2: 0.1, 3: 0.1}, 0.01, "0.1"),
    ],
)
def test_test_selected_adaptive_evidence_uses_checkpoint_objective_through_agent_proposal(
    tmp_path: Path,
    objective_mode: str,
    scores: dict[int, float],
    top_level_score: float,
    expected_score: str,
):
    recipe = _test_selected_adaptive_recipe(tmp_path, objective_mode=objective_mode)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores=scores,
        top_level_score=top_level_score,
    )
    _mark_round_terminal(workflow_dir, tmp_path)

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    digest_row = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")[0]
    assert digest_row["test_auroc"] == expected_score
    assert digest_row["val_ahi_pearson"] == "0.5"
    assert digest_row["checkpoint_path"] == str(checkpoints[2])
    assert digest_row["epoch"] == "2"
    incumbent = _read_table(workflow_dir / "adaptive" / "incumbents.tsv")[-1]
    assert incumbent["run_id"] == run["run_id"]
    assert incumbent["objective_score"] == expected_score
    assert incumbent["checkpoint_path"] == str(checkpoints[2])
    assert incumbent["epoch"] == "2"
    proposal_input = json.loads(input_path.read_text())
    assert proposal_input["input"]["objective"] == {"metric": "test_auroc", "mode": objective_mode}
    proposal_row = proposal_input["input"]["digest_rows"][0]
    assert proposal_row["test_auroc"] == expected_score
    assert proposal_row["checkpoint_path"] == str(checkpoints[2])
    assert proposal_row["epoch"] == "2"


def test_failed_test_checkpoint_objective_stays_unscored_through_agent_proposal(tmp_path: Path):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, _checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.8, 2: 0.9},
        top_level_score=0.99,
    )
    _mark_round_terminal(workflow_dir, tmp_path, status="failed")

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    digest_row = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")[0]
    assert digest_row["status"] == "failed"
    assert digest_row.get("test_auroc", "") == ""
    assert digest_row["checkpoint_path"] == ""
    assert digest_row.get("epoch", "") == ""
    assert not (workflow_dir / "adaptive" / "incumbents.tsv").exists()
    proposal_row = json.loads(input_path.read_text())["input"]["digest_rows"][0]
    assert proposal_row["status"] == "failed"
    assert proposal_row.get("test_auroc", "") == ""
    assert proposal_row["checkpoint_path"] == ""
    assert proposal_row.get("epoch", "") == ""


def test_test_selected_adaptive_evidence_ignores_epoch_checkpoint_symlink(tmp_path: Path):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.8},
        top_level_score=0.99,
    )
    (Path(run["checkpoint_dir"]) / "epoch=2.ckpt").symlink_to(checkpoints[1])
    _mark_round_terminal(workflow_dir, tmp_path)

    digest = adaptive_hparam.digest_hparam_run(workflow_dir)

    row = _read_table(digest)[0]
    assert row["test_auroc"] == "0.8"
    assert row["checkpoint_path"] == str(checkpoints[1])
    assert row["epoch"] == "1"


def test_incomplete_test_checkpoint_evidence_fails_adaptive_reduction(tmp_path: Path):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, _checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.8, 2: 0.9},
        top_level_score=0.99,
    )
    manifest_path = Path(run["runtime_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_test_results"] = manifest["checkpoint_test_results"][:1]
    manifest_path.write_text(json.dumps(manifest))
    _mark_round_terminal(workflow_dir, tmp_path)

    with pytest.raises(ValueError, match="lacks complete checkpoint test evidence"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert not (workflow_dir / "adaptive" / "digests" / "round_000.csv").exists()
    assert not (workflow_dir / "adaptive" / "incumbents.tsv").exists()
    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()


@pytest.mark.parametrize("objective_metric", ["test_auroc", "best_model_score"])
def test_unavailable_completed_adaptive_evidence_fails_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective_metric: str,
):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["objective_metric"] = objective_metric
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, _checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.8},
        top_level_score=0.99,
    )
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "finished",
                "target": "ssh",
                "host": "unit-host",
            }
        ],
    )
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)

    with pytest.raises(ValueError, match="unavailable runtime artifact evidence"):
        adaptive_hparam.digest_hparam_run(workflow_dir)

    assert not (workflow_dir / "adaptive" / "digests" / "round_000.csv").exists()
    assert not (workflow_dir / "adaptive" / "incumbents.tsv").exists()


@pytest.mark.parametrize("objective_value", [None, float("nan")])
def test_completed_adaptive_run_requires_finite_run_level_objective(
    tmp_path: Path,
    objective_value: float | None,
):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["objective_metric"] = "best_model_score"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, _checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.8},
        top_level_score=0.99,
    )
    manifest_path = Path(run["runtime_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if objective_value is None:
        manifest.pop("best_model_score")
    else:
        manifest["best_model_score"] = objective_value
    manifest_path.write_text(json.dumps(manifest))
    _mark_round_terminal(workflow_dir, tmp_path)

    with pytest.raises(ValueError, match="lacks finite best_model_score objective evidence"):
        adaptive_hparam.digest_hparam_run(workflow_dir)

    assert not (workflow_dir / "adaptive" / "digests" / "round_000.csv").exists()
    assert not (workflow_dir / "adaptive" / "incumbents.tsv").exists()


def test_val_selected_adaptive_digest_keeps_top_level_objective_and_validation_checkpoint(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _run_manifest, checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.5, 2: 0.9},
        top_level_score=0.73,
    )

    digest = adaptive_hparam.digest_hparam_run(workflow_dir)

    row = _read_table(digest)[0]
    assert row["test_auroc"] == "0.73"
    assert row["checkpoint_path"] == str(checkpoints[1])
    assert row["epoch"] == "1"


def test_test_selected_adaptive_distinct_test_objective_uses_checkpoint_evidence(tmp_path: Path):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].update({"objective_metric": "test_loss", "objective_mode": "min"})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.5, 2: 0.9, 3: 0.8},
        top_level_score=0.99,
        extra_checkpoint_metrics={"test_loss": {1: 0.4, 2: 0.2, 3: 0.2}},
        extra_top_level_metrics={"test_loss": 0.01},
    )
    _mark_round_terminal(workflow_dir, tmp_path)

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    digest_row = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")[0]
    assert digest_row["test_loss"] == "0.2"
    assert digest_row["checkpoint_path"] == str(checkpoints[2])
    assert digest_row["epoch"] == "2"
    incumbent = _read_table(workflow_dir / "adaptive" / "incumbents.tsv")[-1]
    assert incumbent["run_id"] == run["run_id"]
    assert incumbent["objective_score"] == "0.2"
    assert incumbent["checkpoint_path"] == str(checkpoints[2])
    proposal_row = json.loads(input_path.read_text())["input"]["digest_rows"][0]
    assert proposal_row["test_loss"] == "0.2"
    assert proposal_row["checkpoint_path"] == str(checkpoints[2])


def test_test_selected_adaptive_run_level_objective_keeps_top_level_evidence(tmp_path: Path):
    recipe = _test_selected_adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].update({"objective_metric": "best_model_score", "objective_mode": "max"})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    run, checkpoints = _write_checkpoint_test_manifest(
        workflow_dir,
        scores={1: 0.2, 2: 0.9},
        top_level_score=0.99,
        best_model_score=0.73,
    )
    _mark_round_terminal(workflow_dir, tmp_path)

    input_path = adaptive_hparam.adaptive_step(workflow_dir)

    assert input_path is not None
    digest_row = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")[0]
    assert digest_row["best_model_score"] == "0.73"
    assert digest_row["checkpoint_path"] == str(checkpoints[1])
    assert digest_row["epoch"] == "1"
    incumbent = _read_table(workflow_dir / "adaptive" / "incumbents.tsv")[-1]
    assert incumbent["run_id"] == run["run_id"]
    assert incumbent["objective_score"] == "0.73"
    assert incumbent["checkpoint_path"] == str(checkpoints[1])
    proposal_row = json.loads(input_path.read_text())["input"]["digest_rows"][0]
    assert proposal_row["best_model_score"] == "0.73"
    assert proposal_row["checkpoint_path"] == str(checkpoints[1])


def test_adaptive_digest_uses_canonical_status_not_runtime_manifest(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    runtime_manifest = Path(run["runtime_dir"]) / "run_manifest.json"
    runtime = json.loads(runtime_manifest.read_text())
    runtime["status"] = "completed"
    runtime["metrics"]["status"] = "finished"
    runtime_manifest.write_text(json.dumps(runtime))
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}],
    )
    manifests.write_rows(round_dir / "run_status.tsv", [{**run, "status": "planned"}])

    digest = adaptive_hparam.digest_hparam_run(round_dir)

    assert _read_table(digest)[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "run_status_changed" not in [event["event_type"] for event in events]


def test_adaptive_digest_reads_ssh_artifacts_and_logs_on_the_execution_host(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    workspace = Path(plan["recipe"]["experiment"]["root"])
    merge_run_manifest(
        workspace,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "finished",
                "target": "ssh",
                "host": "unit-host",
                "workdir": "/remote/workdir",
                "gpus": "0",
                "pid_path": "/remote/run.pid",
                "log_path": "/remote/run.log",
                "command": "remote-command",
            }
        ],
    )
    seen = []
    manifest = {
        "best_model_path": "/remote/workdir/log-finetune/version/checkpoints/best-epoch=3.ckpt",
        "metrics": {"test_auroc": 0.73},
    }

    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)
    monkeypatch.setattr(
        run_evidence,
        "runtime_artifacts",
        lambda row: seen.append(("artifacts", row))
        or ("/remote/workdir/log-finetune/version/run_manifest.json", manifest, ["epoch=3.ckpt"]),
    )
    monkeypatch.setattr(
        run_evidence,
        "log_has_failure",
        lambda path, row=None: seen.append(("failed", path, row)) or False,
    )
    monkeypatch.setattr(
        run_evidence,
        "log_tail",
        lambda path, row=None, lines=8: seen.append(("tail", path, row, lines)) or "remote log",
    )

    digest = adaptive_hparam.digest_hparam_run(round_dir)

    row = _read_table(digest)[0]
    assert row["test_auroc"] == "0.73"
    assert row["checkpoint_path"].endswith("/checkpoints/epoch=3.ckpt")
    assert row["run_manifest"] == "/remote/workdir/log-finetune/version/run_manifest.json"
    assert row["log_tail"] == "remote log"
    artifact_row = next(entry[1] for entry in seen if entry[0] == "artifacts")
    assert artifact_row["target"] == "ssh"
    assert artifact_row["host"] == "unit-host"
    log_rows = [entry[2] for entry in seen if entry[0] in {"failed", "tail"}]
    assert all(row["target"] == "ssh" for row in log_rows)
    assert all(row["host"] == "unit-host" for row in log_rows)


def test_adaptive_digest_preflights_outputs_before_monitor(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    digest.parent.mkdir(parents=True)
    digest.symlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()
    monitor_calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "monitor_hparam_runs",
        lambda path: monitor_calls.append(Path(path)) or round_dir / "run_status.tsv",
    )

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.digest_hparam_run(workflow_dir)

    assert monitor_calls == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_adaptive_suggest_preflights_outputs_before_writing(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    suggestion = workflow_dir / "adaptive" / "suggestions" / "round_001.yaml"
    suggestion.parent.mkdir(parents=True)
    suggestion.hardlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_adaptive_suggest_preflights_generated_candidate_before_writing(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    events_before = (tmp_path / "events.jsonl").read_bytes()
    monkeypatch.setattr(adaptive_hparam, "_suggest_parameters", lambda *_args: {"runtime.unsupported": [1]})

    with pytest.raises(RuntimeError, match="Adaptive suggestion failed preflight"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events_before
