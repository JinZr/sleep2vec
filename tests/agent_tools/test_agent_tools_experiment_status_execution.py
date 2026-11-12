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
    _plan, canonical = _add_plan(root, step_id="train", status="lost", task="hparam_tune")
    canonical.update(
        {
            "scheduler_raw_state": "MISSING",
            "scheduler_reason": "Scheduler identity is incomplete.",
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
    }
    assert "schema_version" not in first
    assert "generated_at" not in first
    assert first["summary"] == {"state": "blocked", "run_count": 1, "status_counts": {"lost": 1}}
    assert first["steps"][0]["plan_controller"] == "ordinary"
    assert first["runs"][0]["execution"] == {"target": None}
    assert first["runs"][0]["scheduler"]["raw_state"] == "MISSING"
    assert set(first["runs"][0]["scheduler"]) == {"type", "job_id", "node", "raw_state", "reason"}
    assert first["runs"][0]["process"]["pid"] is None
    assert first["runs"][0]["evidence"]["checkpoint_count"] == "50"
    assert first["blockers"] == [
        {
            "code": "lost",
            "step_id": "train",
            "run_ids": ["run-000"],
            "message": "Scheduler identity is incomplete.",
        }
    ]


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


@pytest.mark.parametrize("status", ["lost", "missing_pid"])
def test_experiment_status_uncertain_states_block_planned_launch(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)
    _add_plan(root, step_id="planned", status="planned")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert [(blocker["code"], blocker["step_id"]) for blocker in snapshot["blockers"]] == [(status, "train")]


@pytest.mark.parametrize("status", ["queued", "launched", "running", "stopping"])
def test_experiment_status_active_states_report_in_progress(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "in_progress"
    assert [(blocker["code"], blocker["run_ids"]) for blocker in snapshot["blockers"]] == [("active_runs", ["run-000"])]


def test_experiment_status_preserves_active_blockers_with_uncertain_runs(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="uncertain", status="lost")
    _add_plan(root, step_id="active", status="running")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert snapshot["summary"]["state"] == "blocked"
    assert blockers_by_step == {"active": ["active_runs"], "uncertain": ["lost"]}


def test_experiment_status_empty_workspace_does_not_guess_a_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"] == {"state": "empty", "run_count": 0, "status_counts": {}}
    assert [blocker["code"] for blocker in snapshot["blockers"]] == ["no_managed_runs"]


def test_experiment_status_reports_ready_to_launch_for_launchable_plans(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune")

    one = experiments.experiment_status(root)

    assert one["summary"]["state"] == "ready_to_launch"
    assert one["blockers"] == []

    _add_plan(root, step_id="train")
    multiple = experiments.experiment_status(root)

    assert multiple["summary"]["state"] == "ready_to_launch"
    assert multiple["blockers"] == []


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
    assert snapshot["blockers"] == [
        {
            "code": "unmaterialized_step",
            "step_id": "evaluate",
            "run_ids": [],
            "message": "The registered step has no materialized plan or canonical runs; experiment-status cannot "
            "prove controller completion.",
        }
    ]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="unmaterialized registered steps"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("status", "state", "code"),
    [("running", "in_progress", "active_runs"), ("lost", "blocked", "lost")],
)
def test_experiment_status_prioritizes_active_hparam_plan_over_ordinary_launch(tmp_path, status, state, code):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="tune", task="hparam_tune", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == state
    assert code in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_attributes_blockers_by_full_managed_run_key(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="first", status="lost")
    _add_plan(root, step_id="second", status="lost")
    _add_plan(root, step_id="terminal", status="failed")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert blockers_by_step["first"] == ["lost"]
    assert blockers_by_step["second"] == ["lost"]
    assert blockers_by_step["terminal"] == []
