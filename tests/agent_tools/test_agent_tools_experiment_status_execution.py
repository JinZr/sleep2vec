from __future__ import annotations

from pathlib import Path

import pytest
from test_agent_tools_experiment_status import _add_plan, _init_workspace, _workspace_files
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
    assert first["runs"][0]["execution"] == {"target": None}
    assert first["runs"][0]["scheduler"]["observed_at"] == "2026-08-25T01:02:03Z"
    assert first["runs"][0]["process"]["pid"] is None
    assert first["runs"][0]["evidence"]["checkpoint_count"] == "50"
    assert first["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert first["decision"]["blocked_actions"] == ["finalize", "launch", "resubmit"]


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


@pytest.mark.parametrize("status", ["unknown_scheduler", "missing_pid", "submitting"])
def test_experiment_status_uncertain_states_block_planned_launch_and_recommend_only_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)
    _add_plan(root, step_id="planned", status="planned")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "launch", "resubmit"]


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
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "launch", "resubmit"]
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
                "plan_controller": "ordinary",
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


@pytest.mark.parametrize(
    ("status", "state", "code"),
    [("running", "in_progress", "active_runs"), ("unknown_scheduler", "blocked", "unknown_scheduler")],
)
def test_experiment_status_prioritizes_active_hparam_plan_over_ordinary_launch(tmp_path, status, state, code):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="tune", task="hparam_tune", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == state
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert code in {blocker["code"] for blocker in snapshot["blockers"]}


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
