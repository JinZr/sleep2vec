from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_tools import adaptive_hparam, hparam_runtime, manifests
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _agent_recipe,
    _mark_round_terminal,
    _write_agent_submission,
    _write_fake_manifest,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


@pytest.fixture
def completed_history(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path, max_rounds=3)
    workflow = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    _write_fake_manifest(workflow, score=0.85)
    _mark_round_terminal(workflow, tmp_path)
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)
    initial_run = json.loads((workflow / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"][0]
    first_input = adaptive_hparam.adaptive_step(workflow)
    assert first_input is not None
    first_proposal = _write_agent_submission(first_input)
    launch_calls = []

    def fake_launch(run_dir, *, dry_run=True):
        launch_calls.append(Path(run_dir))
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        manifests.write_rows(launch_manifest, [{**run, "status": "launched"} for run in runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    adaptive_hparam.adaptive_step(workflow, proposal_path=first_proposal, execute=True)
    second_run = json.loads((workflow / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())["runs"][0]
    checkpoint_dir = Path(second_run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=3.ckpt").write_text("second checkpoint")
    alias = checkpoint_dir / "best-epoch=3.ckpt"
    alias.write_text("second alias")
    manifest = json.loads((Path(initial_run["runtime_dir"]) / "run_manifest.json").read_text())
    manifest.update({"version": second_run["version"], "best_model_path": str(alias)})
    manifest["metrics"]["test_auroc"] = 0.65
    (Path(second_run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    merge_run_manifest(
        tmp_path,
        [{"step_id": second_run["step_id"], "run_id": second_run["run_id"], "status": "finished"}],
    )
    return workflow, initial_run, second_run, first_proposal, launch_calls


def test_next_proposal_keeps_earlier_incumbent_and_can_cite_it_exclusively(tmp_path: Path, completed_history):
    workflow, initial_run, second_run, first_proposal, launch_calls = completed_history
    input_path = adaptive_hparam.adaptive_step(workflow)
    assert input_path is not None
    input_bytes = input_path.read_bytes()
    document = json.loads(input_bytes)
    snapshot = document["input"]
    assert snapshot["source_round"] == 1
    assert snapshot["target_round"] == 2
    rows = snapshot["digest_rows"]
    assert [(row["round"], row["run_id"]) for row in rows] == [
        ("0", initial_run["run_id"]),
        ("1", second_run["run_id"]),
    ]
    assert [float(row["test_auroc"]) for row in rows] == [0.85, 0.65]
    assert [row["run_id"] for row in rows if row["is_incumbent"] == "True"] == [initial_run["run_id"]]
    assert rows[1]["proposal_rationale"] == json.loads(first_proposal.read_text())["rationale"]
    assert rows[1]["proposal_path"] == str(first_proposal)
    assert rows[1]["proposal_sha256"] == adaptive_hparam.file_sha256(first_proposal)
    assert adaptive_hparam.adaptive_step(workflow) == input_path
    assert input_path.read_bytes() == input_bytes

    proposal_path = _write_agent_submission(input_path, lr=[1.5e-6])
    proposal = json.loads(proposal_path.read_text())
    proposal["evidence_run_ids"] = [initial_run["run_id"]]
    proposal["rationale"] = "The earlier run remains best; test a higher learning rate after the lower-rate regression."
    proposal_path.write_text(json.dumps(proposal))
    managed_paths = [
        tmp_path / "events.jsonl",
        tmp_path / "run_manifest.tsv",
        workflow / "adaptive" / "run_registry.tsv",
    ]
    before_preview = {path: path.read_bytes() for path in managed_paths}

    assert adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path) == proposal_path
    assert {path: path.read_bytes() for path in managed_paths} == before_preview
    assert len(launch_calls) == 1

    adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=True)

    accepted = json.loads((workflow / "adaptive" / "proposals" / "round_002.json").read_text())
    assert accepted["evidence_run_ids"] == [initial_run["run_id"]]
    final_runs = json.loads((workflow / "adaptive" / "rounds" / "round_002" / "plan.json").read_text())["runs"]
    assert len(final_runs) == 1
    assert final_runs[0]["runtime.lr"] == 1.5e-6
    assert len(launch_calls) == 2


@pytest.mark.parametrize("changed_evidence", ["earlier_metrics", "accepted_proposal"])
def test_earlier_evidence_drift_blocks_next_proposal_before_publication(
    tmp_path: Path, completed_history, changed_evidence: str
):
    workflow, initial_run, _second_run, first_proposal, launch_calls = completed_history
    input_path = adaptive_hparam.adaptive_step(workflow)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    if changed_evidence == "earlier_metrics":
        changed_path = Path(initial_run["runtime_dir"]) / "run_manifest.json"
        evidence = json.loads(changed_path.read_text())
        evidence["metrics"]["test_auroc"] = 0.55
        expected_error = "current authoritative snapshot"
    else:
        changed_path = first_proposal
        evidence = json.loads(changed_path.read_text())
        evidence["rationale"] = "A changed explanation after this proposal was accepted."
        expected_error = "history differs from accepted round 001"
    changed_path.write_text(json.dumps(evidence))
    managed_paths = [
        tmp_path / "events.jsonl",
        tmp_path / "run_manifest.tsv",
        workflow / "adaptive" / "run_registry.tsv",
        input_path,
        proposal_path,
        changed_path,
    ]
    before = {path: path.read_bytes() for path in managed_paths}

    with pytest.raises(ValueError, match=expected_error):
        adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=True)

    assert {path: path.read_bytes() for path in managed_paths} == before
    assert not (workflow / "adaptive" / "proposals" / "round_002.json").exists()
    assert not (workflow / "adaptive" / "suggestions" / "round_002.yaml").exists()
    assert not (workflow / "adaptive" / "rounds" / "round_002").exists()
    assert len(launch_calls) == 1


@pytest.mark.parametrize("request_already_issued", [False, True])
def test_next_proposal_requires_earlier_round_to_remain_terminal(
    tmp_path: Path, completed_history, request_already_issued: bool
):
    workflow, initial_run, _second_run, _first_proposal, launch_calls = completed_history
    proposal_path = None
    if request_already_issued:
        input_path = adaptive_hparam.adaptive_step(workflow)
        assert input_path is not None
        proposal_path = _write_agent_submission(input_path)
    # Simulate external status drift; ordinary merges preserve terminal outcomes.
    canonical = adaptive_hparam.read_run_manifest(tmp_path)
    next(row for row in canonical if row["run_id"] == initial_run["run_id"])["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", canonical)
    proposal_inputs = workflow / "adaptive" / "proposal_inputs"
    before = {path: path.read_bytes() for path in proposal_inputs.iterdir()}

    with pytest.raises(ValueError, match="history round 000 is not terminal"):
        adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=request_already_issued)

    assert {path: path.read_bytes() for path in proposal_inputs.iterdir()} == before
    assert not (workflow / "adaptive" / "proposals" / "round_002.json").exists()
    assert not (workflow / "adaptive" / "rounds" / "round_002").exists()
    assert len(launch_calls) == 1


def test_proposal_history_excludes_uncommitted_zero_start_round(tmp_path: Path, monkeypatch):
    recipe = _agent_recipe(tmp_path, max_rounds=3)
    workflow = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    _write_fake_manifest(workflow, score=0.85)
    _mark_round_terminal(workflow, tmp_path)
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)
    first_input = adaptive_hparam.adaptive_step(workflow)
    assert first_input is not None
    proposal_path = _write_agent_submission(first_input)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "pending")

    with pytest.raises(RuntimeError, match="started no runs.*was not committed"):
        adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=True)

    registry = test_support._read_table(workflow / "adaptive" / "run_registry.tsv")
    assert {row["round"] for row in registry} == {"0", "1"}
    input_path = adaptive_hparam.adaptive_step(workflow)
    assert input_path is not None
    snapshot = json.loads(input_path.read_text())["input"]
    assert snapshot["source_round"] == 0
    assert snapshot["target_round"] == 2
    assert [(row["round"], row["run_id"]) for row in snapshot["digest_rows"]] == [("0", "run-000")]
    assert snapshot["remaining_budget"]["runs"] == 2
