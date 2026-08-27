from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam, experiments, hparam_runtime, managed_scheduler, manifests
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import _adaptive_recipe, _read_table, _run, _write_fake_manifest

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


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

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
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
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "in_progress"


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


def test_adaptive_round_commit_marker_follows_predecessor_supersede(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    current_run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": current_run["step_id"], "run_id": current_run["run_id"], "status": "pending"}],
    )
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    launches = []

    def fake_launch(run_dir, *, dry_run=True):
        launches.append(Path(run_dir))
        next_runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in next_runs],
        )
        return Path(run_dir) / "launch_manifest.tsv"

    real_append_event = adaptive_hparam._append_event

    def fail_round_commit(root, event_type, payload):
        if event_type == "launch_round":
            raise RuntimeError("round commit marker failed")
        real_append_event(root, event_type, payload)

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "_append_event", fail_round_commit)

    with pytest.raises(RuntimeError, match="round commit marker failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert next(row["status"] for row in rows if row["run_id"] == current_run["run_id"]) == "superseded"
    assert any(row["status"] == "launched" and row["run_id"] != current_run["run_id"] for row in rows)
    assert adaptive_hparam._latest_round_index(workflow_dir) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "supersede_pending_run" in [event["event_type"] for event in events]
    assert "launch_round" not in [event["event_type"] for event in events]

    with pytest.raises(RuntimeError, match="Uncommitted adaptive launch evidence remains"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert launches == [workflow_dir / "adaptive" / "rounds" / "round_001"]
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()


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
        pid_path.write_text(
            json.dumps({"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}) + "\n"
        )
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
    monkeypatch.setattr(hparam_runtime.evidence, "process_identity_running", lambda *_args: True)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", fail_post_start_commit)

    with pytest.raises(RuntimeError, match=r"launch failed.*already committed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    prospective = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == "run-001")
    assert prospective["status"] == "launched"
    assert prospective["target"] == "local"
    assert prospective["pid"] == "123"
    assert prospective["process_group_id"] == "123"
    assert prospective["process_start_token"] == "proc:unit-start"
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
        Path(prospective["pid_path"]).write_text(
            json.dumps({"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}) + "\n"
        )
    elif uncommitted_evidence == "pid_read_error":
        monkeypatch.setattr(
            adaptive_hparam.evidence,
            "read_process_identity",
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


def test_build_failure_retries_the_same_unpublished_round(tmp_path: Path, monkeypatch):
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
    assert not first_attempt.exists()
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert first_attempt.exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_002").exists()
    registry = _read_table(workflow_dir / "adaptive" / "run_registry.tsv")
    assert [row["round"] for row in registry] == ["0", "1"]
    assert adaptive_hparam._latest_round_index(workflow_dir) == 1
    statuses = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert statuses == {"run-000": "superseded", "run-001": "launched"}


def test_round_target_preflight_failure_does_not_advance_registry_or_workspace(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: tmp_path / "digest.csv")
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: recipe)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("target argv rejected")),
    )
    paths = [
        workflow_dir / "adaptive" / "run_registry.tsv",
        workflow_dir / "adaptive" / "workflow.json",
        tmp_path / "run_manifest.tsv",
        tmp_path / "events.jsonl",
    ]
    before = {path: path.read_bytes() for path in paths if path.exists()}

    with pytest.raises(RuntimeError, match="Round 001 plan failed"):
        adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()
    assert not list((workflow_dir / "adaptive" / "rounds").glob(".*.staging"))
    assert {path: path.read_bytes() for path in before} == before


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
