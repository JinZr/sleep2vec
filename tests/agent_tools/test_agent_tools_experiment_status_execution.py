from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent_tools_experiment_status import _add_plan, _init_workspace, _sha256, _workspace_files
from test_agent_tools_experiment_status import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import experiment_tracking, experiments
from agent_tools.manifests import write_rows


def test_experiment_status_snapshot_is_deterministic_and_keeps_recorded_evidence(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train", status="unknown_scheduler", task="hparam_tune")
    canonical.update(
        {
            "scheduler_raw_state": "MISSING",
            "scheduler_reason": "Scheduler identity is incomplete.",
            "scheduler_observed_at": "2026-08-25T01:02:03Z",
            "checkpoint_count": "50",
            "run_manifest": str(root / "runtime" / "train" / "run_manifest.json"),
        }
    )
    write_rows(root / "run_manifest.tsv", [canonical])

    first = experiments.experiment_status(root)
    second = experiments.experiment_status(root)

    assert first == second
    assert set(first) == {
        "experiment",
        "lifecycle_source",
        "live_observation",
        "summary",
        "steps",
        "runs",
        "blockers",
        "decision",
    }
    assert "schema_version" not in first
    assert "generated_at" not in first
    assert first["summary"] == {"state": "blocked", "run_count": 1, "status_counts": {"unknown_scheduler": 1}}
    assert first["steps"][0]["plan_controller"] == "ordinary"
    assert first["runs"][0]["execution"] == {"target": None, "host": None}
    assert first["runs"][0]["scheduler"]["observed_at"] == "2026-08-25T01:02:03Z"
    assert first["runs"][0]["process"]["pid"] is None
    assert first["runs"][0]["evidence"]["checkpoint_count"] == "50"
    assert first["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert first["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]


def test_experiment_status_snapshot_is_independent_of_input_order():
    root = Path("/experiment")
    rows = [
        {"step_id": step_id, "run_id": "run-000", "run_name": "default", "status": "planned"}
        for step_id in ("second", "first")
    ]
    registered_steps = [
        {
            "manifest": {
                "step": {"id": row["step_id"], "phase": "train", "purpose": "Run the fixture."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / row["step_id"]),
                    "task": "finetune",
                    "run_keys": [(row["step_id"], row["run_id"])],
                    "launch_script": str(root / "plans" / row["step_id"] / "run.sh"),
                }
            ],
        }
        for row in rows
    ]
    experiment = {"id": "status-unit", "title": "Status unit"}

    first = experiment_tracking.experiment_status_snapshot(experiment, registered_steps, rows, root=root)
    second = experiment_tracking.experiment_status_snapshot(
        experiment,
        list(reversed(registered_steps)),
        list(reversed(rows)),
        root=root,
    )

    assert first == second


def test_experiment_status_rejects_contradictory_scheduler_identity(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train")
    canonical["scheduler_job_id"] = "123"
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="Direct managed run cannot define Slurm scheduler identity"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("status", ["planned", "pending"])
@pytest.mark.parametrize("field", ["pid", "process_group_id", "process_start_token"])
def test_experiment_status_rejects_process_identity_on_launchable_direct_runs(tmp_path, status, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train", status=status)
    canonical[field] = "42"
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="Launchable direct managed run cannot define PID process identity"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("status", ["unknown_scheduler", "unknown_remote", "missing_pid", "submitting"])
def test_experiment_status_uncertain_states_block_planned_launch_and_recommend_only_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)
    _add_plan(root, step_id="planned", status="planned")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]


@pytest.mark.parametrize("status", ["queued", "launched", "running", "stopping"])
def test_experiment_status_active_states_recommend_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "in_progress"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert "finalize" in snapshot["decision"]["blocked_actions"]
    assert "launch" in snapshot["decision"]["blocked_actions"]


def test_experiment_status_preserves_active_blockers_with_uncertain_runs(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="uncertain", status="unknown_scheduler")
    _add_plan(root, step_id="active", status="running")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]
    assert blockers_by_step == {"active": ["active_runs"], "uncertain": ["unknown_scheduler"]}


def test_experiment_status_empty_workspace_does_not_guess_a_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"] == {"state": "empty", "run_count": 0, "status_counts": {}}
    assert snapshot["blockers"][0]["code"] == "no_managed_runs"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["recommended_next"] is None


def test_experiment_status_launch_actions_and_manual_choice(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    hparam_plan, _row = _add_plan(root, step_id="tune", task="hparam_tune")

    one = experiments.experiment_status(root)

    assert one["summary"]["state"] == "ready_to_launch"
    assert one["decision"]["recommended_next"]["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(hparam_plan),
        "--execute",
    ]

    generic_plan, _row = _add_plan(root, step_id="train")
    multiple = experiments.experiment_status(root)

    assert multiple["decision"]["manual_choice_required"] is True
    assert multiple["decision"]["recommended_next"] is None
    assert [action["id"] for action in multiple["decision"]["other_legal_actions"]] == [
        "run-plan",
        "hparam-run-queue",
    ]
    assert multiple["decision"]["other_legal_actions"][0]["argv"] == ["bash", str(generic_plan / "run.sh")]


@pytest.mark.parametrize(
    ("adaptive", "pipeline", "code"),
    [(True, False, "adaptive_phase_deferred"), (False, True, "pipeline_phase_deferred")],
)
def test_experiment_status_does_not_infer_adaptive_or_pipeline_launch(tmp_path, adaptive, pipeline, code):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(
        root,
        step_id="train",
        task="hparam_tune" if adaptive else "finetune",
        adaptive=adaptive,
        pipeline=pipeline,
    )

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["blockers"][0]["code"] == code
    assert "v1" not in snapshot["blockers"][0]["message"]


def test_experiment_status_rejects_coherent_adaptive_recipe_downgrade(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune", adaptive=True)
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"].pop("adaptive")
    plan["recipe"]["_local_recipe"].pop("adaptive")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved.pop("adaptive")
    resolved["_local_recipe"].pop("adaptive")
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="controller differs from its frozen recipe"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


def test_experiment_status_rejects_pipeline_identity_downgrade(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan_dir, canonical = _add_plan(root, step_id="evaluate", pipeline=True)
    for field in ("pipeline_id", "job_id", "attempt", "result_root"):
        canonical.pop(field)
    write_rows(root / "run_manifest.tsv", [canonical])
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="pipeline run identity is incomplete"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


def test_experiment_status_rejects_missing_step_controller_without_writing(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest.pop("plan_controller")
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="incomplete canonical envelope"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


def test_experiment_status_scopes_deferred_plans_away_from_ordinary_launch(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    ordinary, _row = _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status="completed", adaptive=True)
    _add_plan(root, step_id="pipeline", status="failed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(ordinary / "run.sh")]
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "pipeline_advance"]
    assert {(blocker["code"], blocker["step_id"]) for blocker in snapshot["blockers"]} == {
        ("adaptive_phase_deferred", "adaptive"),
        ("pipeline_phase_deferred", "pipeline"),
    }


def test_experiment_status_blocks_finalize_for_unmaterialized_registered_step(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary", status="completed")
    step_dir = root / "steps" / "evaluate"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Run external evaluation."},
                "experiment_id": "status-unit",
                "plan_controller": "unassigned",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize"]
    assert snapshot["blockers"] == [
        {
            "code": "unmaterialized_step",
            "step_id": "evaluate",
            "run_ids": [],
            "message": "The registered step has no materialized plan or canonical runs; experiment-status cannot "
            "prove controller completion.",
            "blocked_actions": ["finalize"],
        }
    ]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="unmaterialized registered steps"):
        experiments.experiment_status(root)


def test_experiment_status_keeps_multiple_ordinary_candidates_with_deferred_plan(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    first, _row = _add_plan(root, step_id="first")
    second, _row = _add_plan(root, step_id="second")
    _add_plan(root, step_id="pipeline", status="completed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["recommended_next"] is None
    assert [action["argv"] for action in snapshot["decision"]["other_legal_actions"]] == [
        ["bash", str(first / "run.sh")],
        ["bash", str(second / "run.sh")],
    ]
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]


@pytest.mark.parametrize(
    ("status", "state"),
    [("running", "in_progress"), ("unknown_scheduler", "blocked")],
)
def test_experiment_status_prioritizes_active_deferred_plan_over_ordinary_launch(tmp_path, status, state):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status=status, adaptive=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == state
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert "adaptive_phase_deferred" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_attributes_blockers_by_full_managed_run_key(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="first", status="unknown_scheduler")
    _add_plan(root, step_id="second", status="unknown_scheduler")
    _add_plan(root, step_id="terminal", status="failed")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert blockers_by_step["first"] == ["unknown_scheduler"]
    assert blockers_by_step["second"] == ["unknown_scheduler"]
    assert blockers_by_step["terminal"] == []


def test_experiment_status_defers_terminal_pipeline_finalization(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="failed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]
    assert snapshot["blockers"][0]["code"] == "pipeline_phase_deferred"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans") as exc_info:
        experiments.experiment_status(root)
    assert "v1" not in str(exc_info.value)


def test_experiment_status_rejects_completed_pipeline_with_successful_attempt(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)


def test_experiment_status_defers_terminal_adaptive_finalization(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", status="completed", task="hparam_tune", adaptive=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize"]
    assert snapshot["blockers"][0]["code"] == "adaptive_phase_deferred"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_completed_mixed_ordinary_and_deferred_plans(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary", status="completed")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status="completed", adaptive=True)
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)
