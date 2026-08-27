from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from test_agent_tools_hparam_runtime import (
    _REAL_VALIDATED_EXECUTION_SNAPSHOT,
    _RUNTIME_COMMIT,
    _hparam_recipe,
    _process_identity,
    _read_table,
    _run,
    _write_process_identity,
    _write_runtime_rows,
    write_yaml,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
import yaml

from agent_tools import decision_hparam, hparam_runtime, managed_scheduler, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest


def test_hparam_plan_rejects_gpus_per_run_without_a_physical_pool_before_workspace_creation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "gpus_per_run": 2})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"].pop("devices")
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    doctor = _run("doctor", "--recipe", str(recipe))
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    message = "execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices"
    assert doctor.returncode == 1
    assert message in doctor.stdout
    assert planned.returncode == 1
    assert message in planned.stdout
    assert not plan_dir.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("gpus_per_run", [0, False, 0.5, 1.0, 1.5, "0.5", "1", "1.5"])
def test_hparam_execution_reports_invalid_gpus_per_run(gpus_per_run):
    issues = decision_hparam._hparam_execution_issues(
        {"gpu_pool": [0, 1], "gpus_per_run": gpus_per_run},
        {},
    )

    assert len(issues) == 1
    assert issues[0].field == "execution.gpus_per_run"
    assert issues[0].status.value == "FAIL"
    assert "must be a positive integer" in issues[0].message


@pytest.mark.parametrize("max_concurrent", [True, 1.0, 1.5, "1", 0])
def test_hparam_execution_reports_invalid_max_concurrent(max_concurrent):
    issues = decision_hparam._hparam_execution_issues({"max_concurrent": max_concurrent}, {})

    assert len(issues) == 1
    assert issues[0].field == "execution.max_concurrent"
    assert issues[0].status.value == "FAIL"
    assert "must be a positive integer" in issues[0].message


@pytest.mark.parametrize(
    ("execution", "field"),
    [
        ({"python": ""}, "execution.python"),
        ({"python": "conda run -n exp python"}, "execution.python"),
        ({"python": "~/miniconda/bin/python"}, "execution.python"),
        ({"runtime_commit": "abc123"}, "execution.runtime_commit"),
    ],
)
def test_hparam_execution_rejects_invalid_runtime_identity(execution, field):
    issues = decision_hparam._hparam_execution_issues(execution, {})

    assert len(issues) == 1
    assert issues[0].field == field
    assert issues[0].status.value == "FAIL"


def test_hparam_execution_warns_when_slurm_request_voluntarily_lowers_priority():
    issues = decision_hparam._hparam_execution_issues(
        {
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
                "nice": 100,
                "nodelist": "h20-bj-96",
            }
        },
        {},
    )

    priority_issue = next(issue for issue in issues if issue.field == "execution.scheduler.priority")
    assert priority_issue.status.value == "WARN"
    assert "nice=100 voluntarily lowers priority" in priority_issue.message
    assert "nodelist narrows eligible nodes" in priority_issue.message


def test_hparam_runtime_rejects_gpus_per_run_without_a_physical_pool():
    with pytest.raises(
        ValueError,
        match="execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices",
    ):
        hparam_runtime._gpu_groups({"execution": {"gpus_per_run": 2}})


def test_hparam_launch_defaults_to_one_run_per_gpu_group_and_uses_the_free_group(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1},
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 4
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6, 4e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 2
    assert [row["gpus"] for row in rows] == ["0", "1", "", ""]
    assert [row["status"] for row in rows] == ["launched", "launched", "pending", "pending"]
    assert all(rows[index]["target"] == "" and rows[index]["command"] == "" for index in (2, 3))

    merge_run_manifest(
        tmp_path,
        [{"step_id": rows[1]["step_id"], "run_id": rows[1]["run_id"], "status": "finished"}],
    )
    started.clear()
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 1
    assert "CUDA_VISIBLE_DEVICES=1" in started[0]
    assert [row["gpus"] for row in rows] == ["0", "1", "1", ""]
    assert [row["status"] for row in rows] == ["missing_pid", "finished", "launched", "pending"]


def test_hparam_run_queue_dry_run_returns_after_one_preview(tmp_path: Path, monkeypatch):
    calls = []
    manifest = tmp_path / "launch_manifest.tsv"
    monkeypatch.setattr(
        hparam_runtime,
        "launch_hparam_runs",
        lambda plan_dir, *, dry_run: calls.append((Path(plan_dir), dry_run)) or manifest,
    )
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda _seconds: pytest.fail("dry-run must not sleep"))

    result = hparam_runtime.run_hparam_queue(tmp_path / "plan")

    assert result == manifest
    assert calls == [((tmp_path / "plan").resolve(), True)]


def test_hparam_run_queue_executes_each_wave_until_all_runs_are_terminal(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "max_concurrent": 1})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 3
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    sleeps = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, *, script_commits_terminal_status, health: {
            **previous,
            **observation,
            "status": "finished",
        },
    )
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda seconds: sleeps.append(seconds))

    status_path = hparam_runtime.run_hparam_queue(plan_dir, dry_run=False, poll_seconds=0.25)

    assert status_path == plan_dir / "run_status.tsv"
    assert len(started) == 3
    assert sleeps == [0.25, 0.25, 0.25]
    assert {row["status"] for row in _read_table(status_path)} == {"finished"}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event_type"] == "run_status_changed" for event in events) == 3


def test_hparam_run_queue_returns_terminal_plan_without_monitor_or_launch(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(tmp_path, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "finished"}])
    assert not (plan_dir / "run_status.tsv").exists()
    monkeypatch.setattr(hparam_runtime, "monitor_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no monitor"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))

    assert hparam_runtime.run_hparam_queue(plan_dir, dry_run=False) == plan_dir / "run_status.tsv"
    assert _read_table(plan_dir / "run_status.tsv")[0]["status"] == "finished"


def test_hparam_run_queue_fails_instead_of_waiting_on_missing_pid(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "missing_pid"}],
    )
    monkeypatch.setattr(hparam_runtime, "monitor_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no monitor"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("no sleep"))

    with pytest.raises(RuntimeError, match="cannot advance.*missing_pid"):
        hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)


def test_hparam_run_queue_blocks_after_unsafe_process_identity(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    _write_process_identity(rows[0]["pid_path"])

    def reject_identity(*_args, **_kwargs):
        raise run_evidence.ProcessIdentityError("PID 123 was reused by a different process")

    monkeypatch.setattr(run_evidence, "process_identity_running", reject_identity)
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("no sleep"))

    with pytest.raises(RuntimeError, match="cannot advance.*missing_pid"):
        hparam_runtime.run_hparam_queue(tmp_path, dry_run=False)

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "missing_pid"
    assert "reused by a different process" in canonical["process_identity_error"]
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "missing_pid"


def test_hparam_run_queue_fails_after_unbound_remote_launch_remains_unknown(tmp_path: Path, monkeypatch):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "target": "ssh", "host": "unit-host", "status": "launched"}],
    )
    observations = []
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, **_kwargs: observations.append(observation)
        or {**previous, **observation, "status": "unknown_remote"},
    )
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("no sleep"))

    with pytest.raises(RuntimeError, match="unknown_remote without complete process identity"):
        hparam_runtime.run_hparam_queue(tmp_path, dry_run=False)

    assert len(observations) == 1
    row = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert row["status"] == "unknown_remote"
    assert all(row.get(field, "") == "" for field in ("pid", "process_group_id", "process_start_token"))


def test_hparam_run_queue_keeps_monitoring_bound_unknown_remote(tmp_path: Path, monkeypatch):
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "target": "ssh",
                "host": "unit-host",
                "status": "unknown_remote",
                **_process_identity(),
            }
        ],
    )
    observations = iter(["unknown_remote", "finished"])
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, **_kwargs: {
            **previous,
            **observation,
            "status": next(observations),
        },
    )
    launch_calls = []
    sleeps = []
    monkeypatch.setattr(
        hparam_runtime,
        "launch_hparam_runs",
        lambda *_args, **_kwargs: launch_calls.append(True) or tmp_path / "launch_manifest.tsv",
    )
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert hparam_runtime.run_hparam_queue(tmp_path, dry_run=False, poll_seconds=1) == tmp_path / "run_status.tsv"
    assert launch_calls == [True]
    assert sleeps == [1]


def test_hparam_run_queue_records_transition_observed_during_launch(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(tmp_path, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}])
    observations = iter(["running", "finished"])
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, *, script_commits_terminal_status, health: {
            **previous,
            **observation,
            "status": next(observations),
        },
    )
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("terminal queue must not sleep"))

    hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    transitions = [event for event in events if event["event_type"] == "run_status_changed"]
    assert [(event["from"], event["to"]) for event in transitions] == [("running", "finished")]


def test_hparam_run_queue_refreshes_capacity_blocker_from_another_plan(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    first_run = json.loads((first_plan / "plan.json").read_text())["runs"][0]

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    second_run = json.loads((second_plan / "plan.json").read_text())["runs"][0]
    started = []
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, *, script_commits_terminal_status, health: {
            **previous,
            **observation,
            "status": "finished" if previous["run_id"] == first_run["run_id"] else previous["status"],
        },
    )
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launch_failed",
    )
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("queue should make progress"))

    hparam_runtime.run_hparam_queue(second_plan, dry_run=False)

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert rows[first_run["run_id"]]["status"] == "finished"
    assert rows[second_run["run_id"]]["status"] == "launch_failed"
    assert len(started) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(
        event["event_type"] == "run_status_changed"
        and event["run_id"] == first_run["run_id"]
        and event["to"] == "finished"
        for event in events
    )


def test_hparam_launch_rejects_partially_executed_plan_without_snapshot(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan.pop("execution_snapshot")
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).unlink()
    calls = []
    monkeypatch.setattr(
        hparam_runtime,
        "_run_execution_command",
        lambda *_args, **_kwargs: calls.append(True) or pytest.fail("started plan must fail before target probing"),
    )
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)

    with pytest.raises(ValueError, match="after a hparam run has started"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == []
    assert not (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).exists()


def test_hparam_launch_blocks_default_gpu_capacity_when_current_active_identity_is_unknown(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1},
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    merge_run_manifest(
        tmp_path,
        [{"step_id": runs[0]["step_id"], "run_id": runs[0]["run_id"], "status": "running"}],
    )
    monkeypatch.setattr(
        hparam_runtime,
        "_validated_execution_snapshot",
        lambda *_args, **_kwargs: pytest.fail("full capacity must not probe"),
    )
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert [row["status"] for row in rows] == ["running", "pending"]
    assert [row["gpus"] for row in rows] == ["", ""]
    assert started == []


def test_hparam_launch_blocks_default_gpu_capacity_when_other_active_identity_is_unknown(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    first_run = json.loads((first_plan / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": first_run["step_id"],
                "run_id": first_run["run_id"],
                "status": "running",
                "target": "local",
            }
        ],
    )

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    row = _read_table(second_plan / "launch_manifest.tsv")[0]
    assert row["status"] == "pending"
    assert row["gpus"] == ""
    assert started == []


def test_hparam_launch_counts_active_gpu_load_from_previous_plan(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(second_plan, dry_run=True)
    assert [row["gpus"] for row in _read_table(second_plan / "launch_manifest.tsv")] == ["0", "1"]
    second_keys = {
        (row["step_id"], row["run_id"]) for row in json.loads((second_plan / "plan.json").read_text())["runs"]
    }
    assert all(
        row["target"] == "" and row["gpus"] == "" and row["command"] == ""
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if (row["step_id"], row["run_id"]) in second_keys
    )
    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 1
    assert "CUDA_VISIBLE_DEVICES=1" in started[0]
    assert [row["gpus"] for row in rows] == ["1", ""]
    assert [row["status"] for row in rows] == ["launched", "pending"]


def test_hparam_launch_full_previous_plan_keeps_replacement_pending(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_payload = yaml.safe_load(first_recipe.read_text())
    first_payload["search"]["max_runs"] = 2
    first_payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    first_recipe.write_text(yaml.safe_dump(first_payload, sort_keys=False))
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 1
    second_payload["search"]["parameters"]["runtime.lr"] = [3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    first_rows = _read_table(first_plan / "launch_manifest.tsv")
    assert [row["gpus"] for row in first_rows] == ["0", "1"]
    started.clear()

    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    row = _read_table(second_plan / "launch_manifest.tsv")[0]
    assert row["status"] == "pending"
    assert row["gpus"] == ""
    assert started == []


def test_hparam_launch_keeps_cpu_only_concurrency_plan_local(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path)}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_payload = yaml.safe_load(first_recipe.read_text())
    first_payload["runtime"] = {"devices": []}
    write_yaml(first_recipe, first_payload)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 1
    assert [row["status"] for row in rows] == ["launched", "pending"]


def test_hparam_launch_explicit_gpu_oversubscription_warns_and_balances_groups(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "workdir": str(tmp_path),
            "gpu_pool": [0, 1],
            "gpus_per_run": 1,
            "max_concurrent": 4,
        },
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 4
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6, 4e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    assert "Status: WARN" in result.stdout
    assert "GPU oversubscription is explicitly enabled" in result.stdout
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 4
    assert [row["gpus"] for row in rows] == ["0", "1", "0", "1"]
    assert {row["status"] for row in rows} == {"launched"}


def test_hparam_launch_explicit_oversubscription_balances_overlapping_previous_group(tmp_path: Path, monkeypatch):
    first_recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 2},
    )
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["execution"].update(
        {
            "workdir": str(tmp_path),
            "gpu_pool": [0, 1, 2],
            "gpus_per_run": 1,
            "max_concurrent": 4,
        }
    )
    second_payload["search"]["max_runs"] = 4
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6, 4e-6, 5e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 3
    assert "CUDA_VISIBLE_DEVICES=2" in started[0]
    assert [row["gpus"] for row in rows] == ["2", "0", "1", ""]
    assert [row["status"] for row in rows] == ["launched", "launched", "launched", "pending"]


@pytest.mark.parametrize(
    ("different_field", "expected_gpus", "expected_statuses"),
    [
        ("host", ["0", "1"], ["launched", "launched"]),
        ("workdir", ["1", ""], ["launched", "pending"]),
        ("local_host", ["1", ""], ["launched", "pending"]),
    ],
)
def test_hparam_launch_scopes_active_gpu_load_by_target_and_ssh_host(
    tmp_path: Path,
    monkeypatch,
    different_field: str,
    expected_gpus: list[str],
    expected_statuses: list[str],
):
    first_execution = {
        "target": "ssh",
        "host": "host-a",
        "workdir": str(tmp_path / "remote-a"),
        "python": sys.executable,
        "runtime_commit": _RUNTIME_COMMIT,
        "gpu_pool": [0, 1],
        "gpus_per_run": 1,
    }
    if different_field == "local_host":
        first_execution["target"] = "local"
        first_execution["host"] = "local-label-a"
    second_execution = dict(first_execution)
    if different_field == "host":
        second_execution["host"] = "host-b"
    elif different_field == "workdir":
        second_execution["workdir"] = str(tmp_path / "remote-b")
    else:
        second_execution["host"] = "local-label-b"
    first_recipe = _hparam_recipe(tmp_path, execution=first_execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["execution"] = second_execution
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def validate_without_remote(root, paths, remote=None):
        if remote is None:
            return real_validate(root, paths)

    started = []
    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_without_remote)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == expected_statuses.count("launched")
    assert [row["gpus"] for row in rows] == expected_gpus
    assert [row["status"] for row in rows] == expected_statuses
    if different_field in {"workdir", "local_host"}:
        assert "CUDA_VISIBLE_DEVICES=1" in started[0]


@pytest.mark.parametrize("max_concurrent", [True, 1.0, 1.5, "1", 0])
def test_managed_scheduler_rejects_invalid_max_concurrent(max_concurrent):
    with pytest.raises(ValueError, match="execution.max_concurrent must be a positive integer"):
        managed_scheduler.capacity_state(
            {"max_concurrent": max_concurrent},
            {},
            {},
            {},
            expected_keys=set(),
        )


def test_managed_scheduler_capacity_balances_around_other_active_runs():
    execution = {"gpu_pool": [0, 1, 2], "gpus_per_run": 1, "max_concurrent": 3}
    expected = {
        ("evaluate", "run-000"): {
            "step_id": "evaluate",
            "run_id": "run-000",
            "status": "planned",
            "gpus": "",
        }
    }
    workspace = {
        **expected,
        ("train", "run-001"): {
            "step_id": "train",
            "run_id": "run-001",
            "status": "running",
            "target": "local",
            "gpus": "0",
        },
    }

    capacity = managed_scheduler.capacity_state(
        execution,
        {},
        expected,
        workspace,
        expected_keys=set(expected),
    )
    allocation = capacity.next_allocation([(0, expected[("evaluate", "run-000")])])

    assert capacity.slots == 2
    assert allocation is not None
    assert allocation[2] == 1


def test_direct_gpu_capacity_does_not_count_active_slurm_allocations():
    execution = {"gpu_pool": [0, 1], "gpus_per_run": 1, "max_concurrent": 2}
    expected = {
        ("direct", "run-000"): {
            "step_id": "direct",
            "run_id": "run-000",
            "status": "planned",
            "gpus": "",
        }
    }
    workspace = {
        **expected,
        ("slurm", "run-001"): {
            "step_id": "slurm",
            "run_id": "run-001",
            "status": "running",
            "scheduler_type": "slurm",
            "target": "local",
            "gpus": "",
        },
    }

    capacity = managed_scheduler.capacity_state(
        execution,
        {},
        expected,
        workspace,
        expected_keys=set(expected),
    )

    assert capacity.slots == 2
    assert capacity.group_loads == [0, 0]


def test_managed_scheduler_ignores_external_missing_pid_when_expected_runs_are_terminal(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [
            {"run_id": "complete", "status": "finished", "gpus": "0"},
            {"run_id": "blocker", "status": "missing_pid", "gpus": "0"},
        ],
    )
    planned = json.loads((tmp_path / "plan.json").read_text())["runs"]
    monkeypatch.setattr(
        managed_scheduler,
        "observe_run",
        lambda _run_dir, _row, previous, **_kwargs: dict(previous),
    )
    hooks = managed_scheduler.SchedulerHooks(
        validated_snapshot=lambda *_args, **_kwargs: (None, False),
        build_command=lambda *_args, **_kwargs: pytest.fail("terminal runs must not build a command"),
        start_process=lambda *_args, **_kwargs: pytest.fail("terminal runs must not start"),
    )

    result = managed_scheduler.launch_managed_runs(
        tmp_path,
        tmp_path,
        [planned[0]],
        {
            "target": "local",
            "workdir": str(tmp_path),
            "gpu_pool": [0],
            "gpus_per_run": 1,
            "max_concurrent": 1,
        },
        {"devices": [0]},
        dry_run=False,
        fail_on_missing_pid_blocker=True,
        hooks=hooks,
    )

    assert result.committed_rows[0]["run_id"] == rows[0]["run_id"]
    assert result.committed_rows[0]["status"] == "finished"


def test_managed_scheduler_row_terminal_owner_overrides_monitor_default(tmp_path: Path, monkeypatch):
    observed = []

    def status_row(_run_dir, _row, previous, *, script_commits_terminal_status, health):
        observed.append((script_commits_terminal_status, health))
        return dict(previous)

    monkeypatch.setattr(managed_scheduler.evidence, "status_row", status_row)
    row = {
        "step_id": "evaluate",
        "run_id": "run-000",
        "status": "running",
        "terminal_status_owner": "script",
    }

    assert managed_scheduler.observe_run(tmp_path, row, row) == row
    assert observed == [(True, False)]


def test_managed_scheduler_observe_run_preserves_artifact_context(tmp_path: Path, monkeypatch):
    row = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])[0]
    runtime_dir = Path(row["runtime_dir"])
    checkpoint_dir = Path(row["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))
    (checkpoint_dir / "epoch=1.ckpt").write_text("checkpoint")
    (checkpoint_dir / "epoch=2.ckpt").write_text("checkpoint")
    (checkpoint_dir / "notes.txt").write_text("not a checkpoint")
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda _row, _pid: "")

    observed = managed_scheduler.observe_run(tmp_path, row, row, health=True)

    assert "runtime_dir" not in run_evidence.RUN_EVIDENCE_FIELDS
    assert "checkpoint_dir" not in run_evidence.RUN_EVIDENCE_FIELDS
    assert observed["run_manifest"] == str(manifest)
    assert observed["checkpoints"] == "epoch=1.ckpt;epoch=2.ckpt"
    assert observed["checkpoint_count"] == 2


def test_managed_scheduler_validates_result_root_against_explicit_output_root(tmp_path: Path):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    runtime_workdir = tmp_path / "immutable-runtime"
    runtime_workdir.mkdir()
    result_root = tmp_path / "pipelines" / "external" / "results" / "job" / "attempt-001"
    plan = json.loads((tmp_path / "plan.json").read_text())
    run = plan["runs"][0]
    run["result_root"] = str(result_root)
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "result_root": str(result_root),
                "terminal_status_owner": "script",
            }
        ],
    )
    starts = []
    hooks = managed_scheduler.SchedulerHooks(
        validated_snapshot=lambda *_args, **_kwargs: (None, False),
        start_process=lambda _execution, command: starts.append(command) or "launched",
    )

    result = managed_scheduler.launch_managed_runs(
        tmp_path,
        tmp_path,
        [run],
        {"target": "local", "workdir": str(runtime_workdir)},
        {},
        dry_run=False,
        default_script_commits_terminal_status=True,
        runtime_output_fields=("result_root",),
        runtime_output_root=tmp_path,
        hooks=hooks,
    )

    assert len(starts) == 1
    assert result.committed_rows[0]["status"] == "launched"
    assert rows[0]["run_id"] == "run-000"
