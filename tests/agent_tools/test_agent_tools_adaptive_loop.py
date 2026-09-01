from __future__ import annotations

import json
from pathlib import Path
import shlex

import pytest
import yaml

from agent_tools import adaptive_hparam, hparam_runtime, manifests, python_programs, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _adaptive_recipe,
    _read_table,
    _run,
    _test_selected_adaptive_recipe,
    _write_fake_manifest,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


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
    pid_path.write_text(
        json.dumps({"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}) + "\n"
    )
    # Keep the synthetic current run active; this test targets replacement ordering, not an OS process probe.
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: True)
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

    def start_with_pid(_execution, command):
        started.append(command)
        pid = 122 + len(started)
        command_parts = shlex.split(command)
        launcher_index = command_parts.index(python_programs.source("managed_scheduler.process_launch"))
        pid_path = Path(command_parts[launcher_index + 3])
        pid_path.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "process_group_id": pid,
                    "process_start_token": f"proc:unit-start-{pid}",
                }
            )
            + "\n"
        )
        return "launched"

    monkeypatch.setattr(hparam_runtime, "_start_process", start_with_pid)
    hparam_runtime.launch_hparam_runs(round_dir, dry_run=False)
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    assert launch["pid"] == "123"
    assert launch["process_group_id"] == "123"
    assert launch["process_start_token"] == "proc:unit-start-123"
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
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
        (Path(run["run_dir"]) / "pid").write_text(
            json.dumps({"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}) + "\n"
        )
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
    monkeypatch.setattr(hparam_runtime.evidence, "process_identity_running", lambda *_args: True)
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
    assert prospective["pid"] == "123"
    assert prospective["process_group_id"] == "123"
    assert prospective["process_start_token"] == "proc:unit-start"
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


def test_adaptive_loop_materializes_source_before_budget_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    observed = []

    def budget_exhausted(_root, effective_recipe):
        observed.append(effective_recipe["evaluation_policy"]["test_after_fit"])
        return True

    monkeypatch.setattr(adaptive_hparam, "_budget_exhausted", budget_exhausted)

    assert adaptive_hparam.adaptive_loop(workflow_dir) == workflow_dir
    assert observed == [True]
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()

    payload["adaptive"]["max_runs_total"] += 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(
        adaptive_hparam,
        "_budget_exhausted",
        lambda *_args, **_kwargs: pytest.fail("budget check must not run"),
    )

    with pytest.raises(ValueError, match="adaptive.max_runs_total"):
        adaptive_hparam.adaptive_loop(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


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

    confirmed = adaptive_hparam._stop_bad_running_runs(
        workflow_dir, round_dir, adaptive_hparam.load_recipe_with_base(recipe)
    )

    assert seen_rows[0][0] == "/remote/run.log"
    assert seen_rows[0][1]["target"] == "ssh"
    assert seen_rows[0][1]["host"] == "baichuan3"
    assert stopped == [(round_dir, "run-000")]
    assert confirmed == []


@pytest.mark.parametrize(("retirement_credit", "expected_requests"), [(0, 0), (1, 1), (2, 2)])
def test_async_slurm_stop_consumes_credit_without_launching_before_confirmation(
    tmp_path: Path, monkeypatch, retirement_credit: int, expected_requests: int
):
    bad_keys = [("train-model", f"run-{index:03d}") for index in range(3)]
    pending_key = ("train-model-round-001", "run-000")
    pending_row = {"step_id": pending_key[0], "run_id": pending_key[1], "status": "pending"}
    bad_rows = [
        {"step_id": step_id, "run_id": run_id, "status": "running", "scheduler_type": "slurm"}
        for step_id, run_id in bad_keys
    ]
    state = adaptive_hparam._ReplacementState(
        next_round=1,
        next_dir=tmp_path / "round_001",
        next_plan_keys={pending_key},
        started_keys=set(),
        launch_failed_keys=set(),
        retirement_credit=retirement_credit,
    )
    stop_calls = []

    def request_stop(*_args, **kwargs):
        run_key = next(iter(kwargs["run_keys"]))
        stop_calls.append(kwargs["run_keys"])
        row = next(item for item in bad_rows if adaptive_hparam.managed_run_key(item) == run_key)
        row.update(
            {
                "status": "stopping",
                "stop_requested_at": "2026-08-21T03:40:00Z",
                "stop_reason": "adaptive replacement",
            }
        )
        return []

    monkeypatch.setattr(adaptive_hparam, "read_run_manifest", lambda _workspace: [pending_row, *bad_rows])
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", request_stop)
    monkeypatch.setattr(
        adaptive_hparam,
        "_launch_with_recovery",
        lambda *_args, **_kwargs: pytest.fail("replacement must wait for scheduler cancellation confirmation"),
    )

    rows = adaptive_hparam._drain_bad_runs(
        tmp_path,
        tmp_path,
        state,
        tmp_path / "round_000",
        {},
        bad_keys,
        [pending_row],
    )

    assert rows == [pending_row]
    assert stop_calls == [{key} for key in bad_keys[:expected_requests]]
    assert state.retirement_credit == 0
    assert state.stopped_run_keys == []
    assert [row["status"] for row in bad_rows] == ["stopping"] * expected_requests + ["running"] * (
        len(bad_rows) - expected_requests
    )


def test_async_slurm_stop_dispatch_error_does_not_request_later_runs(tmp_path: Path, monkeypatch):
    bad_keys = [("train-model", f"run-{index:03d}") for index in range(2)]
    bad_rows = [{"step_id": step_id, "run_id": run_id, "status": "running"} for step_id, run_id in bad_keys]
    state = adaptive_hparam._ReplacementState(
        next_round=1,
        next_dir=tmp_path / "round_001",
        next_plan_keys=set(),
        started_keys=set(),
        launch_failed_keys=set(),
        retirement_credit=2,
        round_committed=True,
    )
    stop_calls = []

    def fail_after_intent(*_args, **kwargs):
        run_key = next(iter(kwargs["run_keys"]))
        stop_calls.append(run_key)
        row = next(item for item in bad_rows if adaptive_hparam.managed_run_key(item) == run_key)
        row.update(
            {
                "status": "stopping",
                "stop_requested_at": "2026-08-21T03:40:00Z",
                "stop_reason": "adaptive replacement",
            }
        )
        raise RuntimeError("scancel failed")

    monkeypatch.setattr(adaptive_hparam, "read_run_manifest", lambda _workspace: bad_rows)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", fail_after_intent)

    with pytest.raises(RuntimeError, match="failed while stopping run-000"):
        adaptive_hparam._drain_bad_runs(
            tmp_path,
            tmp_path,
            state,
            tmp_path / "round_000",
            {},
            bad_keys,
            [],
        )

    assert stop_calls == [bad_keys[0]]
    assert bad_rows[0]["status"] == "stopping"
    assert bad_rows[1]["status"] == "running"
    assert state.retirement_credit == 2
    assert state.stopped_run_keys == []


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


@pytest.mark.parametrize(
    ("evidence_case", "objective_mode", "scores", "top_level_score", "incumbent_score", "expected_bad"),
    [
        ("absent", "max", None, 0.1, 0.73, False),
        ("incomplete", "max", (0.9, 0.8), 0.1, 0.73, False),
        ("complete-good", "max", (0.9, 0.8), 0.1, 0.73, False),
        ("complete-bad", "max", (0.5, 0.6), 0.99, 0.73, False),
        ("complete-good", "min", (0.1, 0.2), 0.99, 0.27, False),
        ("complete-bad", "min", (0.5, 0.4), 0.01, 0.27, False),
        ("log-failure", "max", None, 0.1, 0.73, True),
    ],
)
def test_test_selected_running_replacement_ignores_checkpoint_objective_until_successful_completion(
    tmp_path: Path,
    evidence_case: str,
    objective_mode: str,
    scores: tuple[float, float] | None,
    top_level_score: float,
    incumbent_score: float,
    expected_bad: bool,
):
    recipe = _test_selected_adaptive_recipe(
        tmp_path,
        objective_mode=objective_mode,
        strategy="best_neighborhood",
        max_rounds=1,
    )
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "epoch=1.ckpt", checkpoint_dir / "epoch=2.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_text(checkpoint.name)
    manifest = {"epoch": 2, "metrics": {"test_auroc": top_level_score}}
    if evidence_case in {"incomplete", "complete-good", "complete-bad"}:
        assert scores is not None
        result_count = 1 if evidence_case == "incomplete" else 2
        manifest.update(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": epoch,
                        "metrics": {"test_auroc": score},
                    }
                    for checkpoint, epoch, score in zip(checkpoints[:result_count], (1, 2), scores)
                ],
            }
        )
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    log_path = round_dir / "logs" / "run-000.log"
    log_path.parent.mkdir()
    log_path.write_text("Traceback\nRuntimeError: failed\n" if evidence_case == "log-failure" else "still training\n")
    (workflow_dir / "adaptive" / "incumbents.tsv").write_text(f"objective_score\n{incumbent_score}\n")
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "running",
                "log_path": str(log_path),
                "launched_at": "2000-01-01T00:00:00Z",
            }
        ],
    )

    bad_keys = adaptive_hparam._bad_running_run_keys(
        workflow_dir,
        round_dir,
        adaptive_hparam.load_recipe_with_base(recipe),
    )

    expected = {(run["step_id"], run["run_id"])} if expected_bad else set()
    assert bad_keys == expected


@pytest.mark.parametrize(
    ("objective_metric", "objective_mode", "checkpoint_scores", "top_level_score", "incumbent", "expected_bad"),
    [
        ("test_loss", "min", (0.1, 0.2), 0.9, 0.4, False),
        ("test_loss", "min", (0.5, 0.6), 0.1, 0.4, False),
        ("best_model_score", "max", (0.1, 0.2), 0.9, 0.73, False),
        ("best_model_score", "max", (0.9, 0.8), 0.5, 0.73, True),
    ],
)
def test_test_selected_running_replacement_distinguishes_checkpoint_and_run_level_objectives(
    tmp_path: Path,
    objective_metric: str,
    objective_mode: str,
    checkpoint_scores: tuple[float, float],
    top_level_score: float,
    incumbent: float,
    expected_bad: bool,
):
    recipe = _test_selected_adaptive_recipe(tmp_path, strategy="best_neighborhood", max_rounds=1)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].update({"objective_metric": objective_metric, "objective_mode": objective_mode})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "epoch=1.ckpt", checkpoint_dir / "epoch=2.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_text(checkpoint.name)
    manifest = {
        "epoch": 2,
        "best_model_score": top_level_score if objective_metric == "best_model_score" else 0.5,
        "metrics": {
            "test_auroc": 0.99,
            **({"test_loss": top_level_score} if objective_metric == "test_loss" else {}),
        },
        "test_all_checkpoints_after_fit": True,
        "checkpoint_test_results": [
            {
                "checkpoint_path": str(checkpoint),
                "epoch": epoch,
                "metrics": {
                    "test_auroc": 0.99 - epoch * 0.01,
                    **({"test_loss": score} if objective_metric == "test_loss" else {}),
                },
            }
            for checkpoint, epoch, score in zip(checkpoints, (1, 2), checkpoint_scores)
        ],
    }
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    log_path = round_dir / "logs" / "run-000.log"
    log_path.parent.mkdir()
    log_path.write_text("still training\n")
    (workflow_dir / "adaptive" / "incumbents.tsv").write_text(f"objective_score\n{incumbent}\n")
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": "running",
                "log_path": str(log_path),
                "launched_at": "2000-01-01T00:00:00Z",
            }
        ],
    )

    bad_keys = adaptive_hparam._bad_running_run_keys(
        workflow_dir,
        round_dir,
        adaptive_hparam.load_recipe_with_base(recipe),
    )

    expected = {(run["step_id"], run["run_id"])} if expected_bad else set()
    assert bad_keys == expected
