from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import adaptive_hparam, experiments, hparam_runtime, manifests, plan_hparam, plans, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


@pytest.fixture(autouse=True)
def _stub_execution_snapshot_preflight(monkeypatch):
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", lambda *_args, **_kwargs: (None, False))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _adaptive_recipe(
    tmp_path: Path, *, test_feedback: bool = True, max_rounds: int = 2, relative_base: bool = False
) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "adaptive_tune.yaml",
        {
            "name": "unit_adaptive",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": base.name if relative_base else str(base),
            "execution": {
                "workdir": str(tmp_path / "runtime"),
                "python": sys.executable,
                "runtime_commit": _RUNTIME_COMMIT,
            },
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6], "yaml:/model/head/name": ["classification"]},
            },
            "adaptive": {
                "enabled": True,
                "objective_metric": "test_auroc",
                "objective_mode": "max",
                "test_feedback_for_selection": test_feedback,
                "max_rounds": max_rounds,
                "max_runs_total": 4,
                "round_size": 1,
                "poll_seconds": 1,
                "replacement": {
                    "enabled": True,
                    "allow_running_stop": True,
                    "grace_epochs": 1,
                    "grace_minutes": 1,
                    "kill_margin": 0.05,
                },
                "suggest": {"strategy": "best_neighborhood"},
            },
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "external optimized adaptive", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _agent_recipe(tmp_path: Path, *, max_rounds: int = 2) -> Path:
    recipe = _adaptive_recipe(tmp_path, max_rounds=max_rounds)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["replacement"] = {"enabled": False}
    payload["adaptive"]["suggest"] = {
        "strategy": "agent_proposal",
        "bounds": {"runtime.lr": [5e-7, 2e-6]},
    }
    recipe.write_text(yaml.safe_dump(payload))
    return recipe


def _write_agent_submission(input_path: Path, *, lr: list[float] | None = None) -> Path:
    proposal_input = json.loads(input_path.read_text())
    proposal_path = Path(proposal_input["expected_proposal_path"])
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": proposal_input["request_id"],
                "target_round": proposal_input["input"]["target_round"],
                "parameters": {
                    "runtime.lr": lr or [5e-7],
                    "yaml:/model/head/name": ["classification"],
                },
                "evidence_run_ids": [proposal_input["input"]["digest_rows"][0]["run_id"]],
                "rationale": "The terminal run supports a lower learning rate within the authorized bounds.",
                "proposer": {"agent": "codex", "model": "gpt-5"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return proposal_path


def _write_fake_manifest(workflow_dir: Path, *, score: float = 0.7) -> None:
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launched = _run("hparam-launch", "--plan-dir", str(round_dir))
    assert launched.returncode == 0, launched.stderr
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=3.ckpt").write_text("checkpoint")
    (ckpt_dir / "best-epoch=3.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": run["version"],
                "monitor": "val_ahi_pearson",
                "monitor_mode": "max",
                "best_model_score": 0.5,
                "best_model_path": str(ckpt_dir / "best-epoch=3.ckpt"),
                "epoch": 3,
                "status": "finished",
                "metrics": {"val_ahi_pearson": 0.5, "test_auroc": score},
            }
        )
    )


def _mark_round_terminal(workflow_dir: Path, workspace: Path, *, status: str = "finished") -> None:
    run = json.loads((workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        workspace,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": status}],
    )


def test_adaptive_recipe_requires_explicit_test_feedback_flag(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, test_feedback=False)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "adaptive.test_feedback_for_selection" in result.stdout


def test_adaptive_rejects_removed_run_budget_and_gpu_fields(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_trials"] = payload["search"].pop("max_runs")
    payload["adaptive"]["max_trials_total"] = payload["adaptive"].pop("max_runs_total")
    payload["execution"]["gpus_per_trial"] = 1
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "search.max_trials is no longer supported" in result.stdout
    assert "adaptive.max_trials_total is no longer supported" in result.stdout
    assert "execution.gpus_per_trial is no longer supported" in result.stdout


def test_agent_proposal_waits_for_terminal_round_then_writes_deterministic_snapshot(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))
    assert result.returncode == 0, result.stderr

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
    assert Path(proposal_input["expected_proposal_path"]).name == (
        f"round_001--{proposal_input['request_id'][7:19]}.json"
    )

    assert adaptive_hparam.adaptive_step(workflow_dir) == input_path
    assert input_path.read_bytes() == first_bytes


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


def _write_agent_configuration_submission(input_path: Path) -> Path:
    proposal_input = json.loads(input_path.read_text())
    proposal_path = Path(proposal_input["expected_proposal_path"])
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": proposal_input["request_id"],
                "target_round": proposal_input["input"]["target_round"],
                "configurations": [
                    {"runtime.lr": 5e-7, "yaml:/model/head/name": "classification"},
                    {"runtime.lr": 2e-6, "yaml:/model/head/name": "classification"},
                ],
                "evidence_run_ids": [proposal_input["input"]["digest_rows"][0]["run_id"]],
                "rationale": "Probe both authorized ends of the LR interval as exact points.",
                "proposer": {"agent": "codex", "model": "gpt-5"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return proposal_path


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

    plan_runs = json.loads(
        (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text()
    )["runs"]
    assert len(plan_runs) == 2  # two points, not the 2x1 product per key
    assert [run["runtime.lr"] for run in plan_runs] == [5e-7, 2e-6]

    accepted = json.loads((workflow_dir / "adaptive" / "proposals" / "round_001.json").read_text())
    assert accepted["configurations"] == suggestion_payload["search"]["configurations"]
    assert "parameters" not in accepted
    rationale = (workflow_dir / "adaptive" / "suggestions" / "round_001.md").read_text()
    assert "## Configurations" in rationale
    assert "point 0" in rationale and "point 1" in rationale
    assert "agent_proposal_accepted" in events_path.read_text()


def test_adaptive_init_preflight_leaves_blocked_root_untouched_then_retries(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "adaptive-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["decisions"]["label_name"] = {"value": "ASK_USER", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    blocked = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert blocked.returncode != 0
    assert not workspace.exists()
    payload["decisions"]["label_name"] = {"value": "ahi", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    retry = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert retry.returncode == 0, retry.stderr
    assert (workspace / "adaptive" / "rounds" / "round_000" / "plan.json").exists()


def test_adaptive_init_creates_round_zero_without_modifying_original_recipe(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    before = recipe.read_text()
    workflow_dir = tmp_path / "workflow"

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == before
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert "adaptive_init" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_rounds_keep_frozen_runtime_identity_and_allow_capacity_updates(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    first_plan = json.loads((round_zero / "plan.json").read_text())
    frozen_identity = {field: first_plan["recipe"]["execution"][field] for field in ("python", "runtime_commit")}
    run = first_plan["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["max_concurrent"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(plan_hparam, "repo_summary", lambda: pytest.fail("later rounds must not resolve HEAD again"))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)
    suggested = yaml.safe_load(suggestion.read_text())
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    round_recipe = adaptive_hparam._write_round_recipe(
        adaptive_hparam.load_recipe_with_base(suggestion), suggestion, next_dir, 1
    )
    report = plans.build_plan(recipe_path=round_recipe, output_dir=next_dir)

    assert report.exit_code == 0
    assert suggested["execution"]["max_concurrent"] == 2
    assert {field: suggested["execution"][field] for field in ("python", "runtime_commit")} == frozen_identity
    second_plan = json.loads((next_dir / "plan.json").read_text())
    assert {
        field: second_plan["recipe"]["execution"][field] for field in ("python", "runtime_commit")
    } == frozen_identity


@pytest.mark.parametrize("tamper", ["missing", "extra"])
def test_adaptive_workflow_rejects_invalid_execution_identity_before_suggestion_write(tmp_path: Path, tamper: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    if tamper == "missing":
        workflow["execution_identity"].pop("runtime_commit")
    else:
        workflow["execution_identity"]["max_concurrent"] = 8
    workflow_path.write_text(json.dumps(workflow))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match="frozen execution identity"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("field", "value"),
    [("python", "/other/python"), ("runtime_commit", "b" * 40)],
)
def test_adaptive_source_rejects_frozen_execution_identity_drift_before_suggestion_write(
    tmp_path: Path, field: str, value: str
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=f"execution.{field} differs"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


def test_adaptive_init_initializes_fresh_experiment_root(tmp_path: Path):
    source_dir = tmp_path / "source"
    recipe = _adaptive_recipe(source_dir)
    payload = yaml.safe_load(recipe.read_text())
    workflow_dir = tmp_path / "fresh-workflow"
    payload["experiment"]["root"] = str(workflow_dir)
    recipe.write_text(yaml.safe_dump(payload))

    assert not workflow_dir.exists()
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "experiment.yaml").exists()
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert _read_table(workflow_dir / "run_manifest.tsv")


def test_adaptive_init_preflights_round_recipe_before_workspace_mutation(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_recipe = workflow_dir / "adaptive" / "rounds" / "round_000" / "round_recipe.yaml"
    round_recipe.parent.mkdir(parents=True)
    round_recipe.symlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert not events_path.exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()


def test_adaptive_init_rejects_symlink_root_before_writing(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    real_root = tmp_path / "real-workflow"
    alias_root = tmp_path / "workflow-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(real_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="experiment root must not be a symlink"):
        adaptive_hparam.init_adaptive_workflow(recipe, alias_root)

    assert alias_root.is_symlink()
    assert not real_root.exists()


def test_adaptive_workflow_root_drift_fails_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["root"] = str(tmp_path / "other-workflow")
    workflow_path.write_text(json.dumps(workflow))
    events = (tmp_path / "events.jsonl").read_bytes()

    result = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "workflow root differs" in result.stderr
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events


def test_adaptive_suggest_rejects_source_contract_drift_before_writing(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_run"] = 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_before = (tmp_path / "events.jsonl").read_bytes()

    result = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "search.max_run" in result.stdout + result.stderr
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_adaptive_relative_recipe_locator_fails_before_suggestion_write(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["recipe_path"] = recipe.name
    workflow_path.write_text(json.dumps(workflow))
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}
    monkeypatch.chdir(recipe.parent)

    with pytest.raises(ValueError, match="recipe_path must be absolute"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_legacy_registry_fails_before_monitor_or_digest_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    (workflow_dir / "adaptive" / "trial_registry.tsv").write_text("trial_id\ntrial_000\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "Legacy adaptive registry is read-only" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


def test_adaptive_registry_must_bind_current_round_before_monitor(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry = workflow_dir / "adaptive" / "run_registry.tsv"
    registry.write_text("step_id\trun_id\tround\tround_dir\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "registry is missing the current plan run" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


@pytest.mark.parametrize("registry_fault", ["foreign", "unmanaged", "config_drift"])
def test_adaptive_registry_ownership_fails_before_workflow_mutation(tmp_path: Path, registry_fault: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry = _read_table(registry_path)
    if registry_fault == "foreign":
        registry[0]["experiment_id"] = "foreign-experiment"
    elif registry_fault == "unmanaged":
        registry.append(
            {
                **registry[0],
                "experiment_id": "foreign-experiment",
                "step_id": "foreign-step",
                "run_id": "run-999",
                "version": "foreign-version",
            }
        )
    else:
        registry[0]["config"] = str(tmp_path / "other-config.yaml")
    manifests.write_rows(registry_path, registry)
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="canonical manifest|Frozen run field differs"):
        adaptive_hparam._workflow(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_registry_rejects_header_only_legacy_identity(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_path.write_text("trial_id\tround\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        adaptive_hparam._workflow(workflow_dir)

    assert registry_path.read_text() == "trial_id\tround\n"


def test_adaptive_stop_scan_ignores_header_only_legacy_projection(tmp_path: Path):
    recipe_path = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe_path), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    status_path = round_dir / "run_status.tsv"
    status_path.write_text("trial_id\tstatus\n")
    recipe = adaptive_hparam.load_recipe_with_base(recipe_path)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe)

    assert status_path.read_text() == "trial_id\tstatus\n"


@pytest.mark.parametrize("explicit_strategy", [False, True])
def test_best_neighborhood_default_and_explicit_use_existing_numeric_neighbors(tmp_path: Path, explicit_strategy: bool):
    recipe = _adaptive_recipe(tmp_path)
    if not explicit_strategy:
        payload = yaml.safe_load(recipe.read_text())
        payload["adaptive"].pop("suggest")
        recipe.write_text(yaml.safe_dump(payload))
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

    with pytest.raises(ValueError, match="Managed output"):
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


def test_adaptive_step_dry_run_writes_suggestion_without_superseding_current_round(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    manifests.write_rows(
        round_dir / "launch_manifest.tsv",
        [{**launch, "status": "planned"}],
    )

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    events = (tmp_path / "events.jsonl").read_text()
    assert "supersede_pending_run" not in events
    assert "adaptive_step_dry_run" in events
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "planned"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "planned"


def test_execute_supersedes_canonical_pending_run_and_prevents_old_round_launch(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"}],
    )
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    launched_rounds = []
    old_status_at_launch = []

    def fake_launch(run_dir, *, dry_run=True):
        launched_rounds.append((Path(run_dir), dry_run))
        old_status_at_launch.append(
            next(row["status"] for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
        )
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in next_runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in next_runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "superseded"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "superseded"
    launch_after = _read_table(round_dir / "launch_manifest.tsv")[0]
    assert launch_after["status"] == "superseded"
    assert launched_rounds == [(workflow_dir / "adaptive" / "rounds" / "round_001", False)]
    assert old_status_at_launch == ["pending"]
    events_path = tmp_path / "events.jsonl"
    before = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in before].count("supersede_pending_run") == 1

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    after = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in after].count("supersede_pending_run") == 1
    started = []
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: started.append(True) or "launched")

    hparam_runtime.launch_hparam_runs(round_dir, dry_run=False)

    assert started == []


def test_supersede_uses_canonical_status_and_repairs_stale_round_mirrors(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}],
    )
    stale = [{**run, "status": "planned", "target": "local", "pid_path": "", "log_path": ""}]
    manifests.write_rows(round_dir / "run_status.tsv", stale)
    manifests.write_rows(round_dir / "launch_manifest.tsv", stale)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "failed"
    assert events_path.read_bytes() == before


def test_supersede_event_uses_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows):
        real_merge(root, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", merge_after_wandb_update)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "failed"
    assert events_path.read_bytes() == before


def test_supersede_does_not_override_run_launched_after_eligibility_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    real_merge = merge_run_manifest

    def merge_after_launch(root, rows):
        real_merge(root, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}])
        return real_merge(root, rows)

    monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", merge_after_launch)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "running"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "running"
    assert events_path.read_bytes() == before


def test_supersede_preflights_round_mirrors_before_canonical_commit(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    mirrors = [{**run, "status": "planned", "target": "local", "pid_path": "", "log_path": ""}]
    manifests.write_rows(round_dir / "run_status.tsv", mirrors)
    manifests.write_rows(round_dir / "launch_manifest.tsv", mirrors)
    target = round_dir / "run_status.tsv"
    target.unlink()
    target.hardlink_to(tmp_path / "run_manifest.tsv")
    manifest_path = tmp_path / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert manifest_path.read_bytes() == before


def test_adaptive_step_blocks_suggestion_without_scored_objective(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    assert _run("hparam-launch", "--plan-dir", str(round_dir)).returncode == 0

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode != 0
    assert "No digest rows with finite test_auroc" in result.stderr
    assert "suggest_blocked" in (tmp_path / "events.jsonl").read_text()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


def test_adaptive_step_execute_resolves_relative_base_recipe_for_next_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3, relative_base=True)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    launched = []

    def fake_launch(run_dir, *, dry_run=True):
        launched.append((Path(run_dir), dry_run))
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in next_runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in next_runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert Path(suggestion["base_recipe"]).is_absolute()
    assert (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "superseded"
    assert (
        _read_table(workflow_dir / "adaptive" / "rounds" / "round_000" / "run_status.tsv")[0]["status"] == "superseded"
    )
    assert (
        _read_table(workflow_dir / "adaptive" / "rounds" / "round_000" / "launch_manifest.tsv")[0]["status"]
        == "superseded"
    )
    assert launched == [(workflow_dir / "adaptive" / "rounds" / "round_001", False)]


def test_adaptive_step_preflights_next_round_before_stop_or_supersede(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    invalid = tmp_path / "invalid-next-round.yaml"
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 0
    invalid.write_text(yaml.safe_dump(payload))
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    digest.parent.mkdir(parents=True)
    digest.write_text("run_id,test_auroc\nrun-000,0.7\n")
    calls = []

    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: invalid)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))

    for execute in (False, True):
        try:
            adaptive_hparam.adaptive_step(workflow_dir, execute=execute)
        except RuntimeError as exc:
            assert "failed preflight" in str(exc)
        else:
            raise AssertionError("adaptive_step should fail before mutating the active round")

        assert calls == []
        assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


def test_adaptive_step_rejects_source_contract_drift_before_digest(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_run"] = 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_before = (tmp_path / "events.jsonl").read_bytes()
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda *_args: calls.append("digest"))
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda *_args: calls.append("suggest"))

    with pytest.raises(RuntimeError, match="Adaptive source recipe failed preflight"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=False)

    assert calls == []
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


@pytest.mark.parametrize("failure_stage", ["build", "registry", "launch"])
def test_adaptive_step_keeps_current_runs_when_replacement_stage_raises(
    tmp_path: Path, monkeypatch, failure_stage: str
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"}],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))

    if failure_stage == "build":
        monkeypatch.setattr(
            adaptive_hparam,
            "build_plan",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
        )
    elif failure_stage == "registry":
        monkeypatch.setattr(
            adaptive_hparam,
            "_append_registry_rows",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("registry failed")),
        )
    else:
        monkeypatch.setattr(
            adaptive_hparam,
            "launch_hparam_runs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    old = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert old["status"] == "pending"
    assert calls == []
    if failure_stage != "build":
        next_runs = json.loads((workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())["runs"]
        next_ids = {row["run_id"] for row in next_runs}
        assert {row["status"] for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] in next_ids} == {
            "planned"
        }
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "launch_round" not in [event["event_type"] for event in events]


def test_adaptive_step_commits_canonical_start_when_initial_launcher_raises(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": current_run["step_id"], "run_id": current_run["run_id"], "status": "pending"}],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))

    def launch_then_raise(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in next_runs],
        )
        raise RuntimeError("launch report failed")

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", launch_then_raise)

    with pytest.raises(
        RuntimeError,
        match=r"launch failed.*already committed.*Superseded current pending runs: run-000",
    ):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert next(row["status"] for row in rows if row["run_id"] == current_run["run_id"]) == "superseded"
    assert any(row["status"] == "launched" and row["run_id"] != current_run["run_id"] for row in rows)
    assert calls == ["launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("launch_round") == 1


def test_adaptive_step_reconciles_pid_after_initial_post_start_commit_failure(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    starts = []

    def start_with_pid(_execution, _command):
        starts.append(True)
        run = json.loads((next_dir / "plan.json").read_text())["runs"][0]
        pid_path = Path(run["run_dir"]) / "pid"
        pid_path.write_text(str(os.getpid()))
        return "launched"

    real_runtime_merge = hparam_runtime.merge_run_manifest
    merge_calls = 0

    def fail_post_start_commit(*args, **kwargs):
        nonlocal merge_calls
        merge_calls += 1
        if merge_calls == 2:
            raise RuntimeError("post-start canonical commit failed")
        return real_runtime_merge(*args, **kwargs)

    monkeypatch.setattr(hparam_runtime, "_start_process", start_with_pid)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", fail_post_start_commit)

    with pytest.raises(RuntimeError, match=r"launch failed.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    prospective = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    assert prospective["status"] == "launched"
    assert prospective["target"] == "local"
    assert prospective["pid"] == str(os.getpid())
    assert prospective["pid_path"] == str(Path(prospective["run_dir"]) / "pid")
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    assert (
        next(row["status"] for row in _read_table(next_dir / "launch_manifest.tsv") if row["run_id"] == "run-001")
        == "launched"
    )
    assert (
        next(row["status"] for row in _read_table(next_dir / "run_status.tsv") if row["run_id"] == "run-001")
        == "launched"
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_launched") == 1

    hparam_runtime.monitor_hparam_runs(next_dir)

    monitored = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    assert monitored["status"] == "running"


@pytest.mark.parametrize("recovery_failure", ["canonical", "mirrors"])
def test_adaptive_step_blocks_retry_when_post_start_reconciliation_fails(
    tmp_path: Path, monkeypatch, recovery_failure: str
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    digest_calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "digest_hparam_run",
        lambda round_dir: digest_calls.append(Path(round_dir)) or tmp_path / "digest.csv",
    )
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    starts = []

    def start_with_pid(_execution, _command):
        starts.append(True)
        run = json.loads((next_dir / "plan.json").read_text())["runs"][0]
        (Path(run["run_dir"]) / "pid").write_text(str(os.getpid()))
        return "launched"

    real_runtime_merge = hparam_runtime.merge_run_manifest
    runtime_merge_calls = 0

    def fail_post_start_commit(*args, **kwargs):
        nonlocal runtime_merge_calls
        runtime_merge_calls += 1
        if runtime_merge_calls == 2:
            raise RuntimeError("post-start canonical commit failed")
        return real_runtime_merge(*args, **kwargs)

    real_adaptive_merge = adaptive_hparam.merge_run_manifest

    def fail_reconciliation(root, rows, **kwargs):
        if any(row.get("status") == "launched" and row.get("pid") for row in rows):
            raise RuntimeError("canonical reconciliation failed")
        return real_adaptive_merge(root, rows, **kwargs)

    monkeypatch.setattr(hparam_runtime, "_start_process", start_with_pid)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", fail_post_start_commit)
    if recovery_failure == "canonical":
        monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", fail_reconciliation)
        error = "launch evidence could not be committed"
        expected_status = "planned"
    else:
        monkeypatch.setattr(
            adaptive_hparam.hparam_runtime,
            "reconcile_hparam_launch_artifacts",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mirror reconciliation failed")),
        )
        error = "launch mirrors or events could not be reconciled"
        expected_status = "launched"

    with pytest.raises(RuntimeError, match=error):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    prospective = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    assert prospective["status"] == expected_status
    assert prospective["target"] == "local"
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    digest_calls.clear()

    assert adaptive_hparam.adaptive_step(workflow_dir, execute=False) == recipe
    assert digest_calls == [workflow_dir / "adaptive" / "rounds" / "round_000"]
    digest_calls.clear()

    with pytest.raises(RuntimeError, match="Uncommitted adaptive launch evidence remains"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert digest_calls == []
    assert starts == [True]
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    assert (
        next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")["status"]
        == expected_status
    )


@pytest.mark.parametrize("launch_status", ["launch_failed", "pending"])
def test_zero_start_replacement_rejects_aliased_round_commit(tmp_path: Path, monkeypatch, launch_status: str):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"}],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))

    def fake_launch(run_dir, *, dry_run=True):
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": launch_status} for row in next_runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": launch_status} for row in next_runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    expected = (
        r"launch failed for .*was not committed"
        if launch_status == "launch_failed"
        else rf"started no runs \(statuses: {launch_status}\)"
    )
    with pytest.raises(RuntimeError, match=expected):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    old = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert old["status"] == "pending"
    prospective = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] != run["run_id"])
    assert prospective["status"] == launch_status
    assert calls == []
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    assert len(_read_table(workflow_dir / "adaptive" / "run_registry.tsv")) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "launch_round" not in [event["event_type"] for event in events]
    next_round_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    forged_events = workflow_dir / "forged-events.jsonl"
    forged_events.write_text(
        json.dumps({"event_type": "launch_round", "round": 1, "round_dir": str(next_round_dir)}) + "\n"
    )
    events_path = tmp_path / "events.jsonl"
    events_path.unlink()
    events_path.symlink_to(forged_events)

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam._workflow(workflow_dir)


@pytest.mark.parametrize(
    ("first_status", "abandoned_status"),
    [("pending", "superseded"), ("launch_failed", "launch_failed")],
)
def test_zero_start_replacement_uses_a_fresh_round_on_the_next_step(
    tmp_path: Path, monkeypatch, first_status: str, abandoned_status: str
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    launch_statuses = iter([first_status, "launched"])
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: next(launch_statuses))

    expected_error = (
        r"started no runs.*was not committed" if first_status == "pending" else r"launch failed.*was not committed"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    first_attempt = workflow_dir / "adaptive" / "rounds" / "round_001"
    assert first_attempt.exists()
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    first_attempt_bytes = {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    }

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    second_attempt = workflow_dir / "adaptive" / "rounds" / "round_002"
    assert second_attempt.exists()
    assert adaptive_hparam._latest_round_index(workflow_dir) == 2
    assert {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    } == first_attempt_bytes
    registry = _read_table(workflow_dir / "adaptive" / "run_registry.tsv")
    assert [row["round_dir"] for row in registry] == [
        str(workflow_dir / "adaptive" / "rounds" / "round_000"),
        str(first_attempt),
        str(second_attempt),
    ]
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "superseded", "run-001": abandoned_status, "run-002": "launched"}


def test_superseded_abandoned_run_still_consumes_registered_run_budget(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["max_runs_total"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "pending")

    with pytest.raises(RuntimeError, match=r"started no runs.*was not committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert adaptive_hparam.adaptive_step(workflow_dir, execute=True) == recipe
    registry = _read_table(workflow_dir / "adaptive" / "run_registry.tsv")
    assert [row["round"] for row in registry] == ["0", "1"]
    assert len(registry) == 2
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "planned", "run-001": "superseded"}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "adaptive_budget_exhausted"


@pytest.mark.parametrize(
    "uncommitted_evidence",
    ["launch_failed_pid", "pid_read_error", "uncertain_status", "failed_status"],
)
def test_adaptive_step_blocks_uncommitted_execution_evidence(tmp_path: Path, monkeypatch, uncommitted_evidence: str):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    digest_calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "digest_hparam_run",
        lambda round_dir: digest_calls.append(Path(round_dir)) or tmp_path / "digest.csv",
    )
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    if uncommitted_evidence in {"launch_failed_pid", "pid_read_error"}:
        monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launch_failed")
        error = "launch failed for run-001"
    else:

        def fail_with_terminal_status(run_dir, *, dry_run=True):
            run = json.loads((Path(run_dir) / "plan.json").read_text())["runs"][0]
            merge_run_manifest(
                tmp_path,
                [
                    {
                        "step_id": run["step_id"],
                        "run_id": run["run_id"],
                        "status": "unknown_remote" if uncommitted_evidence == "uncertain_status" else "failed",
                    }
                ],
            )
            raise RuntimeError("launcher failed after execution observation")

        monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fail_with_terminal_status)
        error = "launch failed"

    with pytest.raises(RuntimeError, match=error):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    prospective = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    if uncommitted_evidence == "launch_failed_pid":
        Path(prospective["pid_path"]).write_text(str(os.getpid()))
    elif uncommitted_evidence == "pid_read_error":
        monkeypatch.setattr(
            adaptive_hparam.evidence,
            "read_pid",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("PID read uncertain")),
        )
    digest_calls.clear()

    with pytest.raises(RuntimeError, match="Uncommitted adaptive launch evidence remains"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert digest_calls == []
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()


def test_adaptive_step_blocks_uncommitted_active_status(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    digest_calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "digest_hparam_run",
        lambda round_dir: digest_calls.append(Path(round_dir)) or tmp_path / "digest.csv",
    )
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_append_registry_rows",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("registry failed")),
    )

    with pytest.raises(RuntimeError, match="registry failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    abandoned = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    merge_run_manifest(
        tmp_path,
        [{"step_id": abandoned["step_id"], "run_id": abandoned["run_id"], "status": "running"}],
    )
    digest_calls.clear()

    with pytest.raises(RuntimeError, match="Uncommitted adaptive launch evidence remains"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert digest_calls == []
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()


def test_build_failure_uses_a_fresh_round_on_the_next_step(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    build_plan = adaptive_hparam.build_plan
    build_calls = 0

    def fail_first_build(**kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            build_plan(**kwargs)
            raise RuntimeError("build failed")
        return build_plan(**kwargs)

    monkeypatch.setattr(adaptive_hparam, "build_plan", fail_first_build)

    with pytest.raises(RuntimeError, match="build failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    first_attempt = workflow_dir / "adaptive" / "rounds" / "round_001"
    first_attempt_bytes = {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    }
    assert _read_table(tmp_path / "run_manifest.tsv")[1]["status"] == "planned"

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    assert {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    } == first_attempt_bytes
    registry = _read_table(workflow_dir / "adaptive" / "run_registry.tsv")
    assert [row["round"] for row in registry] == ["0", "2"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 2
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "superseded", "run-001": "superseded", "run-002": "launched"}


def test_registry_failure_preserves_the_plan_and_next_step_uses_a_fresh_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    append_registry = adaptive_hparam._append_registry_rows
    append_calls = 0

    def fail_first_append(*args):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            raise RuntimeError("registry failed")
        return append_registry(*args)

    monkeypatch.setattr(adaptive_hparam, "_append_registry_rows", fail_first_append)

    with pytest.raises(RuntimeError, match="registry failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    first_attempt = workflow_dir / "adaptive" / "rounds" / "round_001"
    first_attempt_bytes = {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    }
    assert _read_table(tmp_path / "run_manifest.tsv")[1]["status"] == "planned"

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    second_attempt = workflow_dir / "adaptive" / "rounds" / "round_002"
    assert second_attempt.exists()
    assert adaptive_hparam._latest_round_index(workflow_dir) == 2
    assert {
        path.relative_to(first_attempt): path.read_bytes() for path in first_attempt.rglob("*") if path.is_file()
    } == first_attempt_bytes
    registry = _read_table(workflow_dir / "adaptive" / "run_registry.tsv")
    assert [row["round"] for row in registry] == ["0", "2"]
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "superseded", "run-001": "superseded", "run-002": "launched"}
    launched = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-002")
    merge_run_manifest(
        tmp_path,
        [{"step_id": launched["step_id"], "run_id": launched["run_id"], "status": "completed"}],
    )
    report = tmp_path / "final-report.md"
    report.write_text("# Final\n\nAdaptive tuning completed.\n")

    assert experiments.finalize_experiment(tmp_path, report) == tmp_path / "reports" / "final.md"


def test_abandoned_supersede_race_blocks_before_fresh_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    append_registry = adaptive_hparam._append_registry_rows
    append_calls = 0

    def fail_first_append(*args):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            raise RuntimeError("registry failed")
        return append_registry(*args)

    monkeypatch.setattr(adaptive_hparam, "_append_registry_rows", fail_first_append)
    with pytest.raises(RuntimeError, match="registry failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    abandoned = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    real_merge = merge_run_manifest

    def merge_after_launch(root, rows):
        real_merge(
            root,
            [{"step_id": abandoned["step_id"], "run_id": abandoned["run_id"], "status": "running"}],
        )
        return real_merge(root, rows)

    monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", merge_after_launch)

    with pytest.raises(RuntimeError, match="state changed before supersede"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    assert (
        next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")["status"]
        == "running"
    )


@pytest.mark.parametrize("launcher_raises", [False, True])
def test_adaptive_step_mixed_initial_launch_failure_commits_the_live_replacement(
    tmp_path: Path, monkeypatch, launcher_raises: bool
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_runs = json.loads((round_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"} for run in current_runs],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(run) for run in current_runs},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run["run_id"],
                    "status": "launch_failed" if index == 0 else "launched",
                }
                for index, run in enumerate(next_runs)
            ],
        )
        if launcher_raises:
            raise RuntimeError("launch report failed")
        return Path(run_dir) / "launch_manifest.tsv"

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    with pytest.raises(RuntimeError, match=r"launch failed.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    current_by_id = {
        row["run_id"]: row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {"run-000", "run-001"}
    }
    assert [current_by_id[f"run-{index:03d}"]["status"] for index in range(2)] == ["running", "running"]
    assert calls == ["launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    prospective = [row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] not in current_by_id]
    assert {row["status"] for row in prospective} == {"launch_failed", "launched"}


def test_hparam_count_does_not_materialize_search_values():
    recipe = {
        "search": {
            "parameters": {
                "runtime.lr": range(1000),
                "runtime.batch_size": range(1000),
            }
        }
    }

    assert adaptive_hparam._hparam_count(recipe) == 1_000_000


def test_hparam_count_uses_configuration_point_count():
    recipe = {
        "search": {
            "configurations": [
                {"runtime.lr": 1e-6, "runtime.batch_size": 8},
                {"runtime.lr": 2e-6, "runtime.batch_size": 16},
            ],
        }
    }

    assert adaptive_hparam._hparam_count(recipe) == 2


def test_hparam_combos_expands_configuration_points_exactly_and_truncates_by_max_runs():
    points = [
        {"runtime.lr": 1e-6, "yaml:/model/head/name": "classification"},
        {"runtime.lr": 2e-6, "yaml:/model/head/name": "regression"},
    ]

    combos = plan_hparam.hparam_combos({"search": {"configurations": points, "max_runs": 2}})
    assert combos == points
    assert combos[0] is not points[0]  # defensive copy

    truncated = plan_hparam.hparam_combos({"search": {"configurations": points, "max_runs": 1}})
    assert truncated == points[:1]


def test_has_yaml_search_overrides_sees_configuration_point_keys():
    assert plan_hparam.has_yaml_search_overrides(
        {"search": {"configurations": [{"yaml:/model/dim": 128}]}}
    )
    assert not plan_hparam.has_yaml_search_overrides(
        {"search": {"configurations": [{"runtime.lr": 1e-6}]}}
    )


@pytest.mark.parametrize("strategy", ["best_neighborhood", "agent_proposal"])
def test_adaptive_source_recipe_rejects_search_configurations(tmp_path: Path, strategy: str):
    recipe_path = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe_path.read_text())
    payload["search"] = {
        "method": "grid",
        "max_runs": 1,
        "configurations": [{"runtime.lr": 1e-6}],
    }
    payload["adaptive"]["suggest"] = {"strategy": strategy}
    if strategy == "agent_proposal":
        payload["adaptive"].pop("replacement", None)
    write_yaml(recipe_path, payload)

    with pytest.raises(ValueError, match="search.parameters, not search.configurations"):
        adaptive_hparam.init_adaptive_workflow(recipe_path, tmp_path / "workflow")


def test_best_neighborhood_step_replaces_bad_running_run_before_round_terminal(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    hparam_runtime.launch_hparam_runs(round_dir, dry_run=False)
    _write_fake_manifest(workflow_dir, score=0.73)
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    pid_path = Path(launch["pid_path"])
    log_path = Path(launch["log_path"])
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    log_path.write_text("Traceback\nRuntimeError: failed\n")
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}],
    )
    stopped = []
    call_order = []
    real_append_event = adaptive_hparam._append_event

    def fake_launch(run_dir, *, dry_run=True):
        call_order.append("launch")
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        manifests.write_rows(launch_manifest, [{**row, "status": "launched"} for row in next_runs])
        merge_run_manifest(
            tmp_path,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"} for row in next_runs],
        )
        return launch_manifest

    def fake_stop(run_dir, run_id, *, reason):
        call_order.append("stop")
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    def record_launch_round(root, event_type, payload):
        if event_type == "launch_round":
            call_order.append("commit")
        real_append_event(root, event_type, payload)

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)
    monkeypatch.setattr(adaptive_hparam, "_append_event", record_launch_round)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert stopped == [(round_dir, "run-000")]
    assert call_order == ["launch", "commit", "stop"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    assert "stop_bad_running_run" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_step_refreshes_terminal_blocker_before_launching_replacement_on_full_gpu(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update({"gpu_pool": [0], "gpus_per_run": 1})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )
    hparam_runtime.launch_hparam_runs(round_dir, dry_run=False)
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    Path(launch["log_path"]).write_text("Traceback\nRuntimeError: failed\n")
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}],
    )
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    stopped = []
    call_order = []
    real_launch = adaptive_hparam.launch_hparam_runs
    real_append_event = adaptive_hparam._append_event

    def record_launch(run_dir, *, dry_run=True):
        result = real_launch(run_dir, dry_run=dry_run)
        next_status = _read_table(Path(run_dir) / "launch_manifest.tsv")[0]["status"]
        call_order.append(f"launch:{next_status}")
        return result

    def fake_stop(run_dir, run_id, *, reason):
        call_order.append(f"stop:{run_id}")
        stopped.append((Path(run_dir), run_id, reason))
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    def record_event(root, event_type, payload):
        if event_type == "launch_round":
            call_order.append("launch_round")
        real_append_event(root, event_type, payload)

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", record_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)
    monkeypatch.setattr(adaptive_hparam, "_append_event", record_event)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    next_row = _read_table(next_dir / "launch_manifest.tsv")[0]
    assert next_row["status"] == "launched"
    assert next_row["gpus"] == "0"
    assert len(started) == 2
    assert stopped == []
    assert call_order == ["launch:launched", "launch_round"]
    current_row = next(
        row
        for row in adaptive_hparam.read_run_manifest(tmp_path)
        if (row["step_id"], row["run_id"]) == (run["step_id"], run["run_id"])
    )
    assert current_row["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert "stop_bad_running_run" not in event_types
    assert event_types.index("run_status_changed") < event_types.index("launch_round")


@pytest.mark.parametrize("raise_after_drain", [False, True])
def test_adaptive_step_zero_start_after_drain_keeps_old_round_authoritative(
    tmp_path: Path, monkeypatch, raise_after_drain: bool
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_runs = json.loads((round_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"} for run in current_runs],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(run) for run in current_runs},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"} for run in next_runs],
        )
        manifests.write_rows(Path(run_dir) / "launch_manifest.tsv", [{**run, "status": "pending"} for run in next_runs])
        if raise_after_drain and calls.count("launch") == 2:
            raise RuntimeError("launch failed after drain")
        return Path(run_dir) / "launch_manifest.tsv"

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        run = next(item for item in current_runs if item["run_id"] == run_id)
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    error = (
        r"launch failed after the stop attempt for run-000.*was not committed"
        if raise_after_drain
        else r"started no additional runs after stopping run-000.*was not committed"
    )
    with pytest.raises(RuntimeError, match=error):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    current_by_id = {
        row["run_id"]: row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {"run-000", "run-001"}
    }
    assert current_by_id["run-000"]["status"] == "stopped"
    assert current_by_id["run-001"]["status"] == "running"
    next_runs = json.loads((workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())["runs"]
    next_statuses = {
        row["run_id"]: row["status"]
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {run["run_id"] for run in next_runs}
    }
    assert set(next_statuses.values()) == {"pending"}
    assert calls == ["launch", "stop:run-000", "launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "launch_round" not in [event["event_type"] for event in events]


@pytest.mark.parametrize("launcher_raises", [False, True])
def test_adaptive_step_mixed_launch_failure_after_drain_stops_no_additional_run(
    tmp_path: Path, monkeypatch, launcher_raises: bool
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_runs = json.loads((round_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"} for run in current_runs],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(run) for run in current_runs},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        statuses = ["pending", "pending"] if calls.count("launch") == 1 else ["launch_failed", "launched"]
        merge_run_manifest(
            tmp_path,
            [
                {"step_id": run["step_id"], "run_id": run["run_id"], "status": statuses[index]}
                for index, run in enumerate(next_runs)
            ],
        )
        if launcher_raises and calls.count("launch") == 2:
            raise RuntimeError("launch report failed")
        return Path(run_dir) / "launch_manifest.tsv"

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        run = next(item for item in current_runs if item["run_id"] == run_id)
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    with pytest.raises(RuntimeError, match=r"launch failed.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    current_by_id = {
        row["run_id"]: row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {"run-000", "run-001"}
    }
    assert [current_by_id[f"run-{index:03d}"]["status"] for index in range(2)] == ["stopped", "running"]
    assert calls == ["launch", "stop:run-000", "launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1


def test_adaptive_step_commits_canonical_start_when_drain_launcher_raises(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": current_run["step_id"], "run_id": current_run["run_id"], "status": "running"}],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(current_run)},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        status = "pending" if calls.count("launch") == 1 else "launched"
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": status} for run in next_runs],
        )
        if status == "launched":
            raise RuntimeError("launch report failed")
        return Path(run_dir) / "launch_manifest.tsv"

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": current_run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    with pytest.raises(RuntimeError, match=r"launch failed after the stop attempt.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert next(row["status"] for row in rows if row["run_id"] == current_run["run_id"]) == "stopped"
    assert any(row["status"] == "launched" and row["run_id"] != current_run["run_id"] for row in rows)
    assert calls == ["launch", "stop:run-000", "launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("launch_round") == 1


def test_adaptive_step_reconciles_pid_after_post_drain_commit_failure(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    current_run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": current_run["step_id"], "run_id": current_run["run_id"], "status": "running"}],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(current_run)},
    )

    def start_with_pid(_execution, _command):
        run = json.loads((next_dir / "plan.json").read_text())["runs"][0]
        (Path(run["run_dir"]) / "pid").write_text(str(os.getpid()))
        return "launched"

    real_runtime_merge = hparam_runtime.merge_run_manifest
    runtime_merge_calls = 0

    def fail_post_start_commit(*args, **kwargs):
        nonlocal runtime_merge_calls
        runtime_merge_calls += 1
        if runtime_merge_calls == 2:
            raise RuntimeError("post-start canonical commit failed")
        return real_runtime_merge(*args, **kwargs)

    def launch_after_drain(run_dir, *, dry_run=True):
        calls.append("launch")
        if calls.count("launch") == 1:
            return Path(run_dir) / "launch_manifest.tsv"
        return hparam_runtime.launch_hparam_runs(run_dir, dry_run=dry_run)

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": current_run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(hparam_runtime, "_start_process", start_with_pid)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", fail_post_start_commit)
    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", launch_after_drain)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    with pytest.raises(RuntimeError, match=r"launch failed after the stop attempt.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert next(row["status"] for row in rows if row["run_id"] == current_run["run_id"]) == "stopped"
    prospective = next(row for row in rows if row["run_id"] != current_run["run_id"])
    assert prospective["status"] == "launched"
    assert prospective["target"] == "local"
    assert prospective["pid"] == str(os.getpid())
    assert calls == ["launch", "stop:run-000", "launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    assert (
        next(
            row["status"]
            for row in _read_table(next_dir / "launch_manifest.tsv")
            if row["run_id"] != current_run["run_id"]
        )
        == "launched"
    )
    assert (
        next(
            row["status"] for row in _read_table(next_dir / "run_status.tsv") if row["run_id"] != current_run["run_id"]
        )
        == "launched"
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_launched") == 1

    hparam_runtime.monitor_hparam_runs(next_dir)

    monitored = next(
        row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] != current_run["run_id"]
    )
    assert monitored["status"] == "running"


def test_adaptive_step_second_handoff_failure_does_not_stop_third_bad_run(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 3
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6]
    payload["adaptive"]["round_size"] = 3
    payload["adaptive"]["max_runs_total"] = 6
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_runs = json.loads((round_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"} for run in current_runs],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(run) for run in current_runs},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        if calls.count("launch") == 1:
            updates = [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"} for run in next_runs]
        elif calls.count("launch") == 2:
            updates = [
                {
                    "step_id": run["step_id"],
                    "run_id": run["run_id"],
                    "status": "launched" if index == 0 else "pending",
                }
                for index, run in enumerate(next_runs)
            ]
        else:
            updates = []
        if updates:
            merge_run_manifest(tmp_path, updates)
        next_keys = {adaptive_hparam.managed_run_key(run) for run in next_runs}
        next_rows = [
            row
            for row in adaptive_hparam.read_run_manifest(tmp_path)
            if adaptive_hparam.managed_run_key(row) in next_keys
        ]
        manifests.write_rows(Path(run_dir) / "launch_manifest.tsv", next_rows)
        return Path(run_dir) / "launch_manifest.tsv"

    def fake_stop(run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        run = next(item for item in current_runs if item["run_id"] == run_id)
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run_id,
                    "status": "stopped",
                    "stopped_at": manifests.utc_now(),
                    "stop_reason": reason,
                }
            ],
        )
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    with pytest.raises(RuntimeError, match=r"started no additional runs after stopping run-001.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    current_by_id = {
        row["run_id"]: row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {"run-000", "run-001", "run-002"}
    }
    assert [current_by_id[f"run-{index:03d}"]["status"] for index in range(3)] == ["stopped", "stopped", "running"]
    assert calls == ["launch", "stop:run-000", "launch", "stop:run-001", "launch"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1


@pytest.mark.parametrize(
    ("commit_stop", "expected_first_status", "expected_confirmed"),
    [(False, "running", "none"), (True, "stopped", "run-000")],
)
def test_adaptive_step_stop_failure_does_not_relaunch_or_stop_another_run(
    tmp_path: Path,
    monkeypatch,
    commit_stop: bool,
    expected_first_status: str,
    expected_confirmed: str,
):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_runs = json.loads((round_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"} for run in current_runs],
    )
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        adaptive_hparam,
        "_bad_running_run_keys",
        lambda *_args: {adaptive_hparam.managed_run_key(run) for run in current_runs},
    )

    def fake_launch(run_dir, *, dry_run=True):
        calls.append("launch")
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"} for run in next_runs],
        )
        return Path(run_dir) / "launch_manifest.tsv"

    def failing_stop(_run_dir, run_id, *, reason):
        calls.append(f"stop:{run_id}")
        if commit_stop:
            run = next(item for item in current_runs if item["run_id"] == run_id)
            merge_run_manifest(
                tmp_path,
                [
                    {
                        "step_id": run["step_id"],
                        "run_id": run_id,
                        "status": "stopped",
                        "stopped_at": manifests.utc_now(),
                        "stop_reason": reason,
                    }
                ],
            )
        raise RuntimeError("stop failed")

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", failing_stop)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"failed while stopping run-000.*was not committed.*"
            rf"Confirmed stopped current runs: {expected_confirmed}"
        ),
    ):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert calls == ["launch", "stop:run-000"]
    current_by_id = {
        row["run_id"]: row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {"run-000", "run-001"}
    }
    assert [current_by_id[f"run-{index:03d}"]["status"] for index in range(2)] == [
        expected_first_status,
        "running",
    ]
    next_runs = json.loads((workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())["runs"]
    next_statuses = {
        row["run_id"]: row["status"]
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {run["run_id"] for run in next_runs}
    }
    assert set(next_statuses.values()) == {"pending"}
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0


def test_adaptive_step_execute_at_budget_keeps_current_runs_unchanged(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    calls = []

    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert calls == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert "adaptive_budget_exhausted" in event_types
    assert "adaptive_step_dry_run" not in event_types
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()


def test_adaptive_step_checks_prospective_round_size_against_run_budget(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["max_runs_total"] = 2
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args, **_kwargs: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))
    registry = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_before = registry.read_bytes()

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert calls == []
    assert registry.read_bytes() == registry_before
    assert len(_read_table(registry)) == 1
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "adaptive_budget_exhausted"


def test_adaptive_loop_stops_when_step_cannot_create_a_budgeted_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    suggestion = workflow_dir / "adaptive" / "suggestions" / "round_001.yaml"
    calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "adaptive_step",
        lambda path, *, execute=False: calls.append((Path(path), execute)) or suggestion,
    )
    monkeypatch.setattr(
        adaptive_hparam.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("loop should stop before polling")),
    )

    result = adaptive_hparam.adaptive_loop(workflow_dir, execute=True)

    assert result == suggestion
    assert calls == [(workflow_dir, True)]


def test_running_stop_passes_remote_status_row_to_failure_log_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
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
                "status": "running",
                "target": "ssh",
                "host": "baichuan3",
                "workdir": "/remote/workdir",
                "gpus": "0",
                "pid_path": "/remote/run.pid",
                "log_path": "/remote/run.log",
                "command": "remote-command",
            }
        ],
    )
    seen_rows = []
    stopped = []

    def fake_log_has_failure(path, row=None):
        seen_rows.append((path, row))
        return True

    def fake_stop(run_dir, run_id, *, reason):
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(run_evidence, "log_has_failure", fake_log_has_failure)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, adaptive_hparam.load_recipe_with_base(recipe))

    assert seen_rows[0][0] == "/remote/run.log"
    assert seen_rows[0][1]["target"] == "ssh"
    assert seen_rows[0][1]["host"] == "baichuan3"
    assert stopped == [(round_dir, "run-000")]


def test_metric_based_running_stop_honors_grace(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    workspace = Path(plan["recipe"]["experiment"]["root"])
    run_dir = Path(run["runtime_dir"])
    run_dir.mkdir(parents=True)
    log_path = round_dir / "logs" / "run-000.log"
    log_path.parent.mkdir()
    log_path.write_text("still training\n")
    (workflow_dir / "adaptive" / "incumbents.tsv").write_text("objective_score\n0.73\n")
    stopped = []

    def fake_stop(run_dir, run_id, *, reason):
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 0, "metrics": {"test_auroc": 0.6}}))
    merge_run_manifest(
        workspace,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "running",
                "target": "local",
                "host": "",
                "workdir": str(tmp_path),
                "gpus": "",
                "pid_path": str(round_dir / "runs" / "run-000" / "pid"),
                "log_path": str(log_path),
                "command": "unit-command",
                "launched_at": manifests.utc_now(),
            }
        ],
    )
    recipe_data = adaptive_hparam.load_recipe_with_base(recipe)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe_data)

    assert stopped == []
    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 2, "metrics": {"test_auroc": 0.6}}))
    merge_run_manifest(
        workspace,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "running",
                "launched_at": "2000-01-01T00:00:00Z",
            }
        ],
    )

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe_data)

    assert stopped == [(round_dir, "run-000")]
