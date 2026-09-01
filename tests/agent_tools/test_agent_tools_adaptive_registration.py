from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import threading

from agent_tool_test_helpers import write_finetune_recipe
import pytest
import yaml

from agent_tools import adaptive_hparam, hparam_runtime, managed_scheduler, manifests, plan_hparam, plans, run_artifacts
from agent_tools.experiment_workspace import append_event, file_sha256, read_run_manifest, read_step_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import _adaptive_recipe, _agent_recipe, _read_table, _run

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def _slurm_scheduler(**updates):
    return {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
        "nice": 0,
        "nodelist": "gpu[1]",
        "direct_controller": False,
        **updates,
    }


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


def test_generic_plan_rejects_adaptive_recipe_before_workspace_mutation(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "adaptive-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(workspace / "plans" / "generic"),
    )

    assert result.returncode == 1
    assert "Adaptive recipes must be initialized with hparam-adaptive-init" in result.stdout
    assert not workspace.exists()


def test_generic_plan_allows_disabled_adaptive_block(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "static-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["adaptive"]["enabled"] = False
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = workspace / "plans" / "static"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "plan.json").exists()
    assert not (workspace / "adaptive" / "workflow.json").exists()


def test_adaptive_init_creates_round_zero_without_modifying_original_recipe(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    before = recipe.read_text()
    workflow_dir = tmp_path / "workflow"

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == before
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    round_plan_path = workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json"
    assert round_plan_path.exists()
    round_plan = json.loads(round_plan_path.read_text())
    assert "--test-after-fit" in round_plan["runs"][0]["command"]
    assert "--no-test-after-fit" not in round_plan["runs"][0]["command"]
    assert (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert "adaptive_init" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_init_rejects_recipe_root_change_after_lock_selection(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    changed_root = tmp_path / "changed-root"
    workflow_dir = changed_root / "workflow"
    original_lock = adaptive_hparam.plan_registration_lock

    @contextmanager
    def mutate_recipe_after_lock(root):
        with original_lock(root):
            payload = yaml.safe_load(recipe.read_text())
            payload["experiment"]["root"] = str(changed_root)
            recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
            yield

    monkeypatch.setattr(adaptive_hparam, "plan_registration_lock", mutate_recipe_after_lock)

    with pytest.raises(ValueError, match="experiment.root changed while acquiring"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not workflow_dir.exists()
    assert not (changed_root / "experiment.yaml").exists()


def test_adaptive_step_rejects_source_workspace_drift(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workflow_dir)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    (workflow_dir / "experiment.yaml").write_text(
        yaml.safe_dump({"experiment": payload["experiment"]}, sort_keys=False)
    )
    (workflow_dir / "run_manifest.tsv").write_text("step_id\trun_id\n")

    with pytest.raises(ValueError, match="differs from the frozen workflow workspace"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


def test_adaptive_workflow_commit_accepts_relative_round_path(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.chdir(tmp_path)

    plan = run_artifacts.read_hparam_plan(Path("workflow/adaptive/rounds/round_000"))

    assert [run["run_id"] for run in plan["runs"]] == ["run-000"]


def test_adaptive_workflow_commit_resolves_symlink_ancestor_before_dot_dot(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    real_parent = tmp_path / "real-parent"
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, real_parent / "fresh-workflow")
    anchor = real_parent / "anchor"
    anchor.mkdir()
    access = tmp_path / "access"
    access.mkdir()
    (access / "hop").symlink_to(anchor, target_is_directory=True)
    round_dir = access / "hop" / ".." / "fresh-workflow" / "adaptive" / "rounds" / "round_000"

    plan = run_artifacts.read_hparam_plan(round_dir)

    assert round_dir.resolve() == workflow_dir / "adaptive" / "rounds" / "round_000"
    assert [run["run_id"] for run in plan["runs"]] == ["run-000"]


@pytest.mark.parametrize("marker_state", ["missing", "root_drift"])
def test_adaptive_workflow_commit_cannot_be_skipped_by_dot_dot_path(tmp_path: Path, marker_state: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    round_dir = workflow_dir / "adaptive" / "rounds" / ".." / "rounds" / "round_000"
    if marker_state == "missing":
        workflow_path.unlink()
        expected_error = FileNotFoundError
        expected_message = "initialization is not committed"
    else:
        workflow = json.loads(workflow_path.read_text())
        workflow["root"] = str(tmp_path / "other-workflow")
        workflow_path.write_text(json.dumps(workflow))
        expected_error = ValueError
        expected_message = "commit marker differs"

    with pytest.raises(expected_error, match=expected_message):
        run_artifacts.read_hparam_plan(round_dir)


def test_adaptive_workflow_commit_rejects_round_symlink(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    alias = tmp_path / "round-alias"
    alias.symlink_to(workflow_dir / "adaptive" / "rounds" / "round_000", target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        run_artifacts.read_hparam_plan(alias)


def test_adaptive_init_materialization_failure_keeps_round_unpublished_and_retries(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    original_write_text = plan_hparam.write_text

    def fail_plan_markdown(path, text, *, executable=False):
        if Path(path).name == "plan.md":
            raise OSError("injected plan materialization failure")
        return original_write_text(path, text, executable=executable)

    monkeypatch.setattr(plan_hparam, "write_text", fail_plan_markdown)

    with pytest.raises(OSError, match="injected plan materialization failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not round_dir.exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not list((tmp_path / "steps").glob("*/step.yaml"))
    assert _read_table(tmp_path / "run_manifest.tsv") == []
    assert not list((workflow_dir / "adaptive" / "rounds").glob(".*.staging"))

    monkeypatch.setattr(plan_hparam, "write_text", original_write_text)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    plan_text = (round_dir / "plan.json").read_text()
    assert ".staging" not in plan_text
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]


def test_adaptive_init_target_preflight_failure_leaves_no_registration(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("target argv rejected")),
    )

    with pytest.raises(RuntimeError, match="Round 000 plan failed"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not (workflow_dir / "adaptive" / "rounds" / "round_000").exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert not list((workflow_dir / "adaptive" / "rounds").glob(".*.staging"))
    assert not (tmp_path / "steps").exists()
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before


@pytest.mark.parametrize("tamper", [False, True])
def test_adaptive_init_recovers_published_round_before_canonical_registration(
    tmp_path: Path, monkeypatch, tamper: bool
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    original_commit = plan_hparam.commit_hparam_plan
    monkeypatch.setattr(
        plan_hparam,
        "commit_hparam_plan",
        lambda _round_dir, **_kwargs: (_ for _ in ()).throw(OSError("injected canonical registration failure")),
    )

    with pytest.raises(OSError, match="injected canonical registration failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").is_file()
    assert (round_dir / "recipe.resolved.yaml").is_file()
    assert (round_dir / "round_recipe.yaml").is_file()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not list((tmp_path / "steps").glob("*/step.yaml"))
    assert _read_table(tmp_path / "run_manifest.tsv") == []
    plan_bytes = (round_dir / "plan.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        hparam_runtime.launch_hparam_runs(round_dir)
    assert not (round_dir / "launch_manifest.tsv").exists()
    monkeypatch.setattr(hparam_runtime, "monitor_hparam_runs", lambda *_args, **_kwargs: pytest.fail("monitored"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("launched"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args, **_kwargs: pytest.fail("slept"))
    with pytest.raises(FileNotFoundError):
        hparam_runtime.run_hparam_queue(round_dir, dry_run=False)

    if tamper:
        (round_dir / "plan.md").write_text("tampered\n")

    monkeypatch.setattr(plan_hparam, "commit_hparam_plan", original_commit)
    original_build_plan = adaptive_hparam.build_plan
    build_calls = 0

    def count_build(**kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build_plan(**kwargs)

    monkeypatch.setattr(adaptive_hparam, "build_plan", count_build)
    if tamper:
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    else:
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").read_bytes() == plan_bytes
    assert build_calls == 1
    if tamper:
        assert _read_table(tmp_path / "run_manifest.tsv") == []
        assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    else:
        assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
        assert (workflow_dir / "adaptive" / "workflow.json").is_file()


def test_concurrent_adaptive_recoveries_publish_single_registration(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    original_commit = plan_hparam.commit_hparam_plan
    monkeypatch.setattr(
        plan_hparam,
        "commit_hparam_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected registration failure")),
    )
    with pytest.raises(OSError, match="injected registration failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    monkeypatch.setattr(plan_hparam, "commit_hparam_plan", original_commit)

    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at
    registry_writes = 0

    def count_registry_write(path, text, expected_sha256, **kwargs):
        nonlocal registry_writes
        if Path(path) == registry_path:
            registry_writes += 1
        return original_replace(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", count_registry_write)
    original_lock = adaptive_hparam.plan_registration_lock
    first_locked = threading.Event()
    second_attempted = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    lock_attempts = 0
    active_holders = 0
    max_active_holders = 0

    @contextmanager
    def observe_lock(root):
        nonlocal lock_attempts, active_holders, max_active_holders
        with state_lock:
            lock_attempts += 1
            if lock_attempts == 2:
                second_attempted.set()
        with original_lock(root):
            with state_lock:
                active_holders += 1
                max_active_holders = max(max_active_holders, active_holders)
                first_holder = not first_locked.is_set()
            if first_holder:
                first_locked.set()
                assert release_first.wait(timeout=10)
            try:
                yield
            finally:
                with state_lock:
                    active_holders -= 1

    monkeypatch.setattr(adaptive_hparam, "plan_registration_lock", observe_lock)
    errors = []

    def recover():
        try:
            adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=recover)
    second = threading.Thread(target=recover)
    first.start()
    assert first_locked.wait(timeout=10)
    second.start()
    assert second_attempted.wait(timeout=10)
    release_first.set()
    first.join(timeout=30)
    second.join(timeout=30)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert max_active_holders == 1
    assert registry_writes == 1
    assert len(read_run_manifest(tmp_path)) == 1
    assert len(_read_table(registry_path)) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1
    assert not list(round_dir.parent.glob(".*.staging"))


def test_adaptive_init_waits_for_ordinary_registration_owner(tmp_path: Path, monkeypatch):
    adaptive_recipe = _adaptive_recipe(tmp_path)
    adaptive_payload = yaml.safe_load(adaptive_recipe.read_text())
    ordinary_recipe = Path(adaptive_payload["base_recipe"])
    adaptive_payload["step"] = yaml.safe_load(ordinary_recipe.read_text())["step"]
    adaptive_recipe.write_text(yaml.safe_dump(adaptive_payload, sort_keys=False))
    ordinary_plan = tmp_path / "plans" / "ordinary"
    workflow_dir = tmp_path / "workflow"
    ordinary_holding = threading.Event()
    release_ordinary = threading.Event()
    adaptive_staging = threading.Event()
    original_check = plans._assert_no_incomplete_step_registration
    original_stage = adaptive_hparam._stage_round

    def pause_ordinary(recipe, out):
        if out == ordinary_plan:
            ordinary_holding.set()
            if not release_ordinary.wait(timeout=10):
                raise AssertionError("ordinary planner was not released")
        return original_check(recipe, out)

    def observe_adaptive_stage(*args, **kwargs):
        adaptive_staging.set()
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(plans, "_assert_no_incomplete_step_registration", pause_ordinary)
    monkeypatch.setattr(adaptive_hparam, "_stage_round", observe_adaptive_stage)
    ordinary_reports = []
    ordinary_errors = []
    adaptive_errors = []

    def run_ordinary():
        try:
            ordinary_reports.append(plans.build_plan(recipe_path=ordinary_recipe, output_dir=ordinary_plan))
        except BaseException as exc:
            ordinary_errors.append(exc)

    def run_adaptive():
        try:
            adaptive_hparam.init_adaptive_workflow(adaptive_recipe, workflow_dir)
        except BaseException as exc:
            adaptive_errors.append(exc)

    ordinary = threading.Thread(target=run_ordinary)
    adaptive = threading.Thread(target=run_adaptive)
    ordinary.start()
    assert ordinary_holding.wait(timeout=10)
    adaptive.start()
    try:
        assert not adaptive_staging.wait(timeout=0.5)
    finally:
        release_ordinary.set()
    ordinary.join(timeout=30)
    adaptive.join(timeout=30)

    assert not ordinary_errors
    assert ordinary_reports[0].exit_code == 0
    assert len(adaptive_errors) == 1
    assert "plan_controller differs" in str(adaptive_errors[0])
    assert len(read_run_manifest(tmp_path)) == 1
    assert read_step_manifest(tmp_path, adaptive_payload["step"]["id"])["plan_controller"] == "ordinary"
    assert not (workflow_dir / "adaptive" / "rounds" / "round_000").exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()


def test_concurrent_fresh_adaptive_initializations_are_idempotent(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    original_stage = adaptive_hparam._stage_round
    staged_dirs = []

    def count_staging(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        staged_dirs.append(staged)
        return staged

    monkeypatch.setattr(adaptive_hparam, "_stage_round", count_staging)
    original_publish = adaptive_hparam.publish_staged_plan_locked
    publications = 0

    def count_publication(*args, **kwargs):
        nonlocal publications
        publications += 1
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(adaptive_hparam, "publish_staged_plan_locked", count_publication)
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at
    registry_writes = 0

    def count_registry_write(path, text, expected_sha256, **kwargs):
        nonlocal registry_writes
        if Path(path) == registry_path:
            registry_writes += 1
        return original_replace(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", count_registry_write)
    results = []
    errors = []

    def initialize():
        try:
            results.append(adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=initialize)
    second = threading.Thread(target=initialize)
    first.start()
    second.start()
    first.join(timeout=60)
    second.join(timeout=60)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [workflow_dir, workflow_dir]
    assert len(staged_dirs) == 1
    assert publications == 1
    assert registry_writes == 1
    assert len(read_run_manifest(tmp_path)) == 1
    assert len(_read_table(registry_path)) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1
    assert not list(round_dir.parent.glob(".*.staging"))
    assert not list(round_dir.parent.glob(".*.backup"))


def test_concurrent_adaptive_init_rechecks_publication_after_preflight(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    original_preflight = adaptive_hparam.preflight_plan
    first_waiting = threading.Event()
    second_preflight = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    preflight_calls = 0

    def pause_first_preflight(**kwargs):
        nonlocal preflight_calls
        assert kwargs["allow_existing_output_artifacts"] is True
        with state_lock:
            preflight_calls += 1
            first_call = preflight_calls == 1
        if first_call:
            first_waiting.set()
            assert release_first.wait(timeout=30)
        else:
            second_preflight.set()
        return original_preflight(**kwargs)

    monkeypatch.setattr(adaptive_hparam, "preflight_plan", pause_first_preflight)
    results = []
    errors = []

    def initialize():
        try:
            results.append(adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=initialize)
    first.start()
    assert first_waiting.wait(timeout=30)
    second = threading.Thread(target=initialize)
    second.start()
    try:
        assert not second_preflight.wait(timeout=0.5)
    finally:
        release_first.set()
    first.join(timeout=60)
    second.join(timeout=60)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [workflow_dir, workflow_dir]
    assert preflight_calls == 2
    assert len(read_run_manifest(tmp_path)) == 1
    assert len(_read_table(workflow_dir / "adaptive" / "run_registry.tsv")) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1
    assert not list((workflow_dir / "adaptive" / "rounds").rglob("*.staging"))


@pytest.mark.parametrize("event_written", [False, True], ids=["before-write", "after-write"])
def test_adaptive_init_reconciles_plan_event_after_append_failure(
    tmp_path: Path,
    monkeypatch,
    event_written: bool,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_write_event = adaptive_hparam._write_experiment_event
    failed = False

    def fail_plan_event(workspace, event_type, payload):
        nonlocal failed
        if event_type == "plan_created" and not failed:
            failed = True
            if event_written:
                original_write_event(workspace, event_type, payload)
            raise OSError("injected plan event failure")
        return original_write_event(workspace, event_type, payload)

    monkeypatch.setattr(adaptive_hparam, "_write_experiment_event", fail_plan_event)
    with pytest.raises(OSError, match="injected plan event failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert len(read_run_manifest(tmp_path)) == 1
    assert not workflow_path.exists()
    monkeypatch.setattr(adaptive_hparam, "_write_experiment_event", original_write_event)

    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1


@pytest.mark.parametrize("event_written", [False, True], ids=["before-write", "after-write"])
def test_adaptive_init_reconciles_ready_event_after_append_failure(
    tmp_path: Path,
    monkeypatch,
    event_written: bool,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_write_event = adaptive_hparam._write_experiment_event
    failed = False

    def fail_ready_event(workspace, event_type, payload):
        nonlocal failed
        if event_type == "adaptive_init" and not failed:
            failed = True
            if event_written:
                original_write_event(workspace, event_type, payload)
            raise OSError("injected adaptive event failure")
        return original_write_event(workspace, event_type, payload)

    monkeypatch.setattr(adaptive_hparam, "_write_experiment_event", fail_ready_event)
    with pytest.raises(OSError, match="injected adaptive event failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert workflow_path.is_file()
    monkeypatch.setattr(adaptive_hparam, "_write_experiment_event", original_write_event)

    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1


def test_adaptive_consumers_reject_marker_before_ready_event(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_reconcile = adaptive_hparam._reconcile_event
    marker_published = threading.Event()
    release_initializer = threading.Event()
    errors = []

    def pause_ready_event(workspace, event_type, payload, *, identity_field):
        if event_type == "adaptive_init":
            marker_published.set()
            assert release_initializer.wait(timeout=30)
        return original_reconcile(
            workspace,
            event_type,
            payload,
            identity_field=identity_field,
        )

    monkeypatch.setattr(adaptive_hparam, "_reconcile_event", pause_ready_event)

    def initialize():
        try:
            adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
        except BaseException as exc:
            errors.append(exc)

    initializer = threading.Thread(target=initialize)
    initializer.start()
    assert marker_published.wait(timeout=30)
    assert workflow_path.is_file()

    with pytest.raises(FileNotFoundError, match="initialization events are not committed"):
        adaptive_hparam.digest_hparam_run(round_dir)

    assert not (workflow_dir / "adaptive" / "digests").exists()
    release_initializer.set()
    initializer.join(timeout=60)
    assert not initializer.is_alive()
    assert errors == []
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert event_types.index("plan_created") < event_types.index("adaptive_init")
    assert "digest" not in event_types


@pytest.mark.parametrize("event_type", ["plan_created", "adaptive_init"])
def test_adaptive_consumers_reject_conflicting_initialization_events(tmp_path: Path, event_type: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    if event_type == "plan_created":
        plan = json.loads((round_dir / "plan.json").read_text())
        payload = {
            "step_id": plan["recipe"]["step"]["id"],
            "plan_dir": str(round_dir),
            "run_count": 99,
        }
    else:
        payload = {
            "round": 0,
            "recipe_path": str(tmp_path / "other-recipe.yaml"),
            "round_dir": str(round_dir),
        }
    append_event(tmp_path, event_type, payload)

    with pytest.raises(ValueError, match="initialization events conflict"):
        run_artifacts.read_hparam_plan(round_dir)
    with pytest.raises(ValueError, match="initialization events conflict"):
        hparam_runtime.launch_hparam_runs(round_dir)
    with pytest.raises(ValueError, match="initialization events conflict"):
        adaptive_hparam.digest_hparam_run(round_dir)

    assert not (round_dir / "launch_manifest.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


def test_adaptive_init_rejects_ready_event_without_plan_event(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events if event["event_type"] != "plan_created")
    )
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match="without its plan-created event"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert events_path.read_bytes() == events_before


def test_adaptive_init_rejects_ready_event_without_workflow_marker(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    events_path = tmp_path / "events.jsonl"
    rows = _read_table(registry_path)
    rows.append(
        {
            **rows[0],
            "round": "1",
            "run_id": "run-001",
            "run_name": "later-round",
            "version": "later-round",
            "round_dir": str(workflow_dir / "adaptive" / "rounds" / "round_001"),
        }
    )
    manifests.write_rows(registry_path, rows)
    workflow_path.unlink()
    events_before = events_path.read_bytes()
    registry_before = registry_path.read_bytes()

    with pytest.raises(ValueError, match="without its workflow marker"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not workflow_path.exists()
    assert events_path.read_bytes() == events_before
    assert registry_path.read_bytes() == registry_before


@pytest.mark.parametrize("conflict", ["plan", "plan_exact", "ready"])
def test_adaptive_init_rejects_preexisting_event_conflicts_before_registration(tmp_path: Path, conflict: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    if conflict in {"plan", "plan_exact"}:
        step_id = adaptive_hparam.load_recipe_with_base(recipe)["step"]["id"]
        event = {
            "event_type": "plan_created",
            "step_id": step_id,
            "plan_dir": str(round_dir),
            "run_count": 99 if conflict == "plan" else 1,
        }
        expected = (
            "event history conflicts: plan_created" if conflict == "plan" else "exists before canonical registration"
        )
    else:
        event = {
            "event_type": "adaptive_init",
            "round": 0,
            "recipe_path": str(recipe.resolve()),
            "round_dir": str(round_dir),
        }
        expected = "without its plan-created event"
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps(event, sort_keys=True) + "\n")
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=expected):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert events_path.read_bytes() == events_before
    assert not round_dir.exists()
    assert not (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert read_run_manifest(tmp_path) == []


@pytest.mark.parametrize(
    ("event_type", "field", "replacement"),
    [
        ("plan_created", "run_count", True),
        ("plan_created", "run_count", 1.0),
        ("adaptive_init", "round", False),
        ("adaptive_init", "round", 0.0),
    ],
    ids=["plan-bool", "plan-float", "ready-bool", "ready-float"],
)
def test_adaptive_init_rejects_event_payload_type_changes(
    tmp_path: Path,
    event_type: str,
    field: str,
    replacement,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    next(event for event in events if event["event_type"] == event_type)[field] = replacement
    events_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=f"event history conflicts: {event_type}"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert events_path.read_bytes() == events_before


def test_adaptive_init_rejects_readme_ancestor_drift_after_parent_open(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    adaptive_dir = workflow_dir / "adaptive"
    moved_adaptive_dir = workflow_dir / "adaptive-moved"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_readme = outside_dir / "README.md"
    outside_readme.write_text("external sentinel\n")
    outside_before = outside_readme.read_bytes()
    original_open_temporary = adaptive_hparam.exp_io._open_temporary_at
    swapped = False

    def swap_readme_ancestor(parent_descriptor, target_name):
        nonlocal swapped
        if target_name == "README.md" and not swapped:
            swapped = True
            adaptive_dir.rename(moved_adaptive_dir)
            adaptive_dir.symlink_to(outside_dir, target_is_directory=True)
        return original_open_temporary(parent_descriptor, target_name)

    monkeypatch.setattr(adaptive_hparam.exp_io, "_open_temporary_at", swap_readme_ancestor)

    with pytest.raises(ValueError, match="Managed CAS path changed during publication"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert swapped is True
    assert outside_readme.read_bytes() == outside_before
    assert not (moved_adaptive_dir / "README.md").exists()
    assert not (outside_dir / "workflow.json").exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert not [event for event in events if event["event_type"] == "adaptive_init"]


def test_adaptive_init_rejects_workflow_ancestor_drift_after_parent_open(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    adaptive_dir = workflow_dir / "adaptive"
    moved_adaptive_dir = workflow_dir / "adaptive-moved"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    original_open_temporary = adaptive_hparam.exp_io._open_temporary_at
    swapped = False

    def swap_workflow_ancestor(parent_descriptor, target_name):
        nonlocal swapped
        if target_name == "workflow.json" and not swapped:
            swapped = True
            adaptive_dir.rename(moved_adaptive_dir)
            adaptive_dir.symlink_to(outside_dir, target_is_directory=True)
        return original_open_temporary(parent_descriptor, target_name)

    monkeypatch.setattr(adaptive_hparam.exp_io, "_open_temporary_at", swap_workflow_ancestor)

    with pytest.raises(ValueError, match="Managed CAS path changed during publication"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not (outside_dir / "workflow.json").exists()
    assert not (moved_adaptive_dir / "workflow.json").exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert not [event for event in events if event["event_type"] == "adaptive_init"]


def test_adaptive_init_recovers_canonical_round_before_workflow_commit(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at

    def fail_workflow_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == workflow_path:
            raise OSError("injected workflow commit failure")
        return original_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", fail_workflow_commit)

    with pytest.raises(OSError, match="injected workflow commit failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    plan_bytes = (round_dir / "plan.json").read_bytes()
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    assert len(_read_table(registry_path)) == 1
    registry_before = registry_path.read_bytes()
    assert not workflow_path.exists()
    with pytest.raises(FileNotFoundError, match="initialization is not committed"):
        hparam_runtime.launch_hparam_runs(round_dir)
    assert not (round_dir / "launch_manifest.tsv").exists()

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", original_replace)
    monkeypatch.setattr(adaptive_hparam, "build_plan", lambda **_kwargs: pytest.fail("retry rebuilt round 000"))
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").read_bytes() == plan_bytes
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
    assert len(_read_table(registry_path)) == 1
    assert registry_path.read_bytes() == registry_before
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1


@pytest.mark.parametrize("registry_bytes", [b"broken\n", b"\xff"], ids=["truncated", "invalid-utf8"])
def test_adaptive_init_repairs_malformed_initial_registry(tmp_path: Path, monkeypatch, registry_bytes: bytes):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at

    def fail_workflow_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == workflow_path:
            raise OSError("injected workflow commit failure")
        return original_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", fail_workflow_commit)
    with pytest.raises(OSError, match="injected workflow commit failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    registry_path.write_bytes(registry_bytes)
    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", original_replace)

    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert workflow_path.is_file()
    assert len(_read_table(registry_path)) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_adaptive_registry_repair_rejects_alias_after_topology_guard(
    tmp_path: Path,
    monkeypatch,
    alias_kind: str,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at

    def fail_workflow_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == workflow_path:
            raise OSError("injected workflow commit failure")
        return original_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", fail_workflow_commit)
    with pytest.raises(OSError, match="injected workflow commit failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", original_replace)
    outside = tmp_path / "outside.tsv"
    outside_bytes = b"external sentinel\n"
    outside.write_bytes(outside_bytes)
    original_validate = adaptive_hparam.exp_io.validate_managed_output_paths
    swapped = False

    def swap_after_guard(root, paths, **kwargs):
        nonlocal swapped
        result = original_validate(root, paths, **kwargs)
        if not swapped and [Path(path) for path in paths] == [registry_path]:
            registry_path.unlink()
            if alias_kind == "symlink":
                registry_path.symlink_to(outside)
            else:
                registry_path.hardlink_to(outside)
            swapped = True
        return result

    monkeypatch.setattr(adaptive_hparam.exp_io, "validate_managed_output_paths", swap_after_guard)

    with pytest.raises(ValueError, match="missing or aliased"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert swapped is True
    assert outside.read_bytes() == outside_bytes
    assert not workflow_path.exists()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_adaptive_init_rejects_registry_alias_before_public_readiness(
    tmp_path: Path,
    monkeypatch,
    alias_kind: str,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    outside = tmp_path / "outside.tsv"
    original_ensure_registry = adaptive_hparam._ensure_initial_registry

    def alias_registry_after_ensure(root, round_dir, plan):
        original_ensure_registry(root, round_dir, plan)
        registry_bytes = registry_path.read_bytes()
        outside.write_bytes(registry_bytes)
        registry_path.unlink()
        if alias_kind == "symlink":
            registry_path.symlink_to(outside)
        else:
            registry_path.hardlink_to(outside)

    monkeypatch.setattr(adaptive_hparam, "_ensure_initial_registry", alias_registry_after_ensure)

    with pytest.raises(ValueError, match="missing or aliased"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not workflow_path.exists()
    assert outside.read_bytes() == registry_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="initialization is not committed"):
        hparam_runtime.launch_hparam_runs(round_dir)
    assert not (round_dir / "launch_manifest.tsv").exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert not [event for event in events if event["event_type"] == "adaptive_init"]


@pytest.mark.parametrize("support_name", ["run_registry.tsv", "README.md"])
def test_adaptive_init_binds_support_files_to_workflow_publication(
    tmp_path: Path,
    monkeypatch,
    support_name: str,
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    adaptive_dir = workflow_dir / "adaptive"
    workflow_path = adaptive_dir / "workflow.json"
    support_path = adaptive_dir / support_name
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at
    drifted = False

    def drift_support_before_workflow_cas(path, text, expected_sha256, **kwargs):
        nonlocal drifted
        if Path(path) == workflow_path and not drifted:
            support_path.write_text(support_path.read_text() + "drift\n")
            drifted = True
        return original_replace(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", drift_support_before_workflow_cas)

    with pytest.raises(RuntimeError, match="inputs changed before readiness publication"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert drifted is True
    assert not workflow_path.exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert not [event for event in events if event["event_type"] == "adaptive_init"]


def test_adaptive_registry_repair_rejects_ancestor_alias_before_cas(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    adaptive_dir = workflow_dir / "adaptive"
    moved_adaptive_dir = workflow_dir / "adaptive-original"
    workflow_path = adaptive_dir / "workflow.json"
    registry_path = adaptive_dir / "run_registry.tsv"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at

    def fail_workflow_commit(path, text, expected_sha256, **kwargs):
        if Path(path) == workflow_path:
            raise OSError("injected workflow commit failure")
        return original_replace(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", fail_workflow_commit)
    with pytest.raises(OSError, match="injected workflow commit failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    registry_path.write_bytes(b"broken\n")
    registry_before = registry_path.read_bytes()
    outside = tmp_path / "outside-adaptive"
    outside.mkdir()
    outside_registry = outside / registry_path.name
    outside_registry.write_bytes(registry_before)
    swapped = False

    def swap_ancestor_before_cas(path, text, expected_sha256, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == registry_path:
            adaptive_dir.rename(moved_adaptive_dir)
            adaptive_dir.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_replace(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", swap_ancestor_before_cas)

    with pytest.raises(OSError):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert swapped is True
    assert outside_registry.read_bytes() == registry_before
    assert not (outside / "workflow.json").exists()


def test_adaptive_init_does_not_repair_valid_registry_mismatch(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_registry_writer = adaptive_hparam._ensure_initial_registry

    def write_invalid_registry(root, initial_round_dir, plan):
        original_registry_writer(root, initial_round_dir, plan)
        registry_path = root / "adaptive" / "run_registry.tsv"
        rows = _read_table(registry_path)
        rows[0]["config"] = str(tmp_path / "other-config.yaml")
        manifests.write_rows(registry_path, rows)

    monkeypatch.setattr(adaptive_hparam, "_ensure_initial_registry", write_invalid_registry)

    with pytest.raises(ValueError, match="Frozen run field differs"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").is_file()
    assert _read_table(tmp_path / "run_manifest.tsv")
    assert not workflow_path.exists()
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_before = registry_path.read_bytes()

    monkeypatch.setattr(adaptive_hparam, "_ensure_initial_registry", original_registry_writer)
    with pytest.raises(ValueError, match="differs from the frozen round"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert registry_path.read_bytes() == registry_before
    assert not workflow_path.exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert not [event for event in events if event["event_type"] == "adaptive_init"]


@pytest.mark.parametrize("missing_name", ["recipe.resolved.yaml", "plan.md"])
def test_adaptive_init_rejects_visible_incomplete_round_without_repair(tmp_path: Path, missing_name: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow_path.unlink()
    missing_path = round_dir / missing_name
    missing_path.unlink()
    plan_before = (round_dir / "plan.json").read_bytes()
    manifest_path = tmp_path / "run_manifest.tsv"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(FileNotFoundError):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not missing_path.exists()
    assert (round_dir / "plan.json").read_bytes() == plan_before
    assert manifest_path.read_bytes() == manifest_before
    assert not workflow_path.exists()


def test_adaptive_rounds_keep_frozen_route_and_python_and_allow_commit_and_capacity_updates(
    tmp_path: Path, monkeypatch
):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow_bytes = workflow_path.read_bytes()
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    first_plan = json.loads((round_zero / "plan.json").read_text())
    frozen_identity = {field: first_plan["recipe"]["execution"][field] for field in ("python", "runtime_commit")}
    frozen_route = adaptive_hparam._execution_route(first_plan["recipe"]["execution"])
    next_commit = "b" * 40
    assert next_commit != frozen_identity["runtime_commit"]
    run = first_plan["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(
        {
            **first_plan["recipe"]["execution"],
            "python": frozen_identity["python"],
            "runtime_commit": next_commit,
            "max_concurrent": 2,
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(plan_hparam, "repo_summary", lambda: pytest.fail("later rounds must not resolve HEAD again"))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)
    suggested = yaml.safe_load(suggestion.read_text())
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    staging_dir = adaptive_hparam._stage_round(
        next_dir,
        adaptive_hparam.load_recipe_with_base(suggestion),
        suggestion,
        1,
        None,
    )
    adaptive_hparam._publish_staged_round_locked(staging_dir, next_dir)
    plan_hparam.commit_hparam_plan(next_dir)

    assert suggested["execution"]["max_concurrent"] == 2
    assert suggested["execution"]["python"] == frozen_identity["python"]
    assert suggested["execution"]["runtime_commit"] == next_commit
    second_plan = json.loads((next_dir / "plan.json").read_text())
    assert second_plan["recipe"]["execution"]["python"] == frozen_identity["python"]
    assert second_plan["recipe"]["execution"]["runtime_commit"] == next_commit
    assert adaptive_hparam._execution_route(second_plan["recipe"]["execution"]) == frozen_route
    assert workflow_path.read_bytes() == workflow_bytes


def test_adaptive_publication_serializes_with_doctor_output(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workspace = tmp_path / "workflow"
    round_dir = workspace / "adaptive" / "rounds" / "round_000"
    doctor_path = write_finetune_recipe(tmp_path / "doctor-source", include_label=False)
    doctor_recipe, _config, doctor_report = plans.evaluate_recipe(doctor_path)
    assert doctor_report.exit_code == 2

    publication_entered = threading.Event()
    allow_publication = threading.Event()
    doctor_attempted = threading.Event()
    publish_staged_round = adaptive_hparam._publish_staged_round_locked
    plan_lock = plans.plan_publication_lock

    def pause_publication(*args, **kwargs):
        publication_entered.set()
        assert allow_publication.wait(timeout=10)
        return publish_staged_round(*args, **kwargs)

    def observe_doctor_lock(out):
        doctor_attempted.set()
        return plan_lock(out)

    monkeypatch.setattr(adaptive_hparam, "_publish_staged_round_locked", pause_publication)
    monkeypatch.setattr(plans, "plan_publication_lock", observe_doctor_lock)
    init_errors = []
    doctor_errors = []

    def initialize():
        try:
            adaptive_hparam.init_adaptive_workflow(recipe, workspace)
        except BaseException as exc:
            init_errors.append(exc)

    def write_doctor():
        try:
            plans.write_doctor_outputs(round_dir, doctor_recipe, doctor_report)
        except BaseException as exc:
            doctor_errors.append(exc)

    init_thread = threading.Thread(target=initialize)
    doctor_thread = threading.Thread(target=write_doctor)
    init_thread.start()
    assert publication_entered.wait(timeout=10)
    doctor_thread.start()
    assert doctor_attempted.wait(timeout=10)
    allow_publication.set()
    init_thread.join(timeout=30)
    doctor_thread.join(timeout=30)

    assert not init_thread.is_alive()
    assert not doctor_thread.is_alive()
    assert init_errors == []
    assert len(doctor_errors) == 1
    assert "Plan artifacts already exist" in str(doctor_errors[0])
    assert (round_dir / "plan.json").exists()
    assert not (round_dir / "decisions.yaml").exists()
    experiment_root = Path(json.loads((round_dir / "plan.json").read_text())["recipe"]["experiment"]["root"])
    assert len(read_run_manifest(experiment_root)) == 1


def test_uncommitted_adaptive_round_restores_bound_config_placeholder(tmp_path: Path):
    round_dir = tmp_path / "round_001"
    round_dir.mkdir()
    bound_config = round_dir / "config.source.yaml"
    bound_config.write_text("source: frozen\n")
    staging_dir = tmp_path / ".round_001.staging"
    staging_dir.mkdir()
    (staging_dir / "plan.json").write_text("{}\n")
    (staging_dir / bound_config.name).write_bytes(bound_config.read_bytes())

    staged_plan_sha256 = run_artifacts.plan_tree_sha256(staging_dir)
    placeholder_backup = adaptive_hparam._publish_staged_round_locked(
        staging_dir,
        round_dir,
        bound_config_path=bound_config,
        bound_config_sha256=file_sha256(bound_config),
    )
    restored = adaptive_hparam._restore_uncommitted_round(
        staging_dir,
        round_dir,
        placeholder_backup,
        staged_plan_sha256,
    )

    assert restored is True
    assert {path.name for path in round_dir.iterdir()} == {bound_config.name}
    assert bound_config.read_text() == "source: frozen\n"
    assert (staging_dir / "plan.json").exists()
    assert placeholder_backup is not None
    assert not placeholder_backup.exists()


def test_uncommitted_adaptive_round_preserves_foreign_bytes_on_failed_restore(tmp_path: Path):
    round_dir = tmp_path / "round_001"
    round_dir.mkdir()
    bound_config = round_dir / "config.source.yaml"
    bound_config.write_text("source: frozen\n")
    staging_dir = tmp_path / ".round_001.staging"
    staging_dir.mkdir()
    (staging_dir / "plan.json").write_text("{}\n")
    (staging_dir / bound_config.name).write_bytes(bound_config.read_bytes())
    staged_plan_sha256 = run_artifacts.plan_tree_sha256(staging_dir)

    placeholder_backup = adaptive_hparam._publish_staged_round_locked(
        staging_dir,
        round_dir,
        bound_config_path=bound_config,
        bound_config_sha256=file_sha256(bound_config),
    )
    foreign = round_dir / "user.txt"
    foreign.write_text("preserve me\n")
    restored = adaptive_hparam._restore_uncommitted_round(
        staging_dir,
        round_dir,
        placeholder_backup,
        staged_plan_sha256,
    )

    assert restored is False
    assert foreign.read_text() == "preserve me\n"
    assert not staging_dir.exists()
    assert placeholder_backup is not None
    assert (placeholder_backup / bound_config.name).read_text() == "source: frozen\n"


def test_adaptive_init_preserves_staging_when_rollback_raises(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workspace = tmp_path / "workflow"
    preserved = []

    def fail_validation(*_args, **_kwargs):
        raise ValueError("injected final validation failure")

    def fail_restore(staging_dir, round_dir, _placeholder_backup, _staged_plan_sha256):
        round_dir.replace(staging_dir)
        (staging_dir / "user.txt").write_text("preserve me\n")
        preserved.append(staging_dir)
        raise OSError("injected rollback failure")

    monkeypatch.setattr(adaptive_hparam, "_validate_initial_round", fail_validation)
    monkeypatch.setattr(adaptive_hparam, "_restore_uncommitted_round", fail_restore)

    with pytest.raises(OSError, match="injected rollback failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workspace)

    assert len(preserved) == 1
    assert (preserved[0] / "user.txt").read_text() == "preserve me\n"


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


def test_adaptive_workflow_rejects_baseline_commit_drift_from_round_zero(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["execution_identity"]["runtime_commit"] = "b" * 40
    workflow_path.write_text(json.dumps(workflow))

    with pytest.raises(ValueError, match="baseline execution identity differs from round 000"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()


def test_adaptive_source_rejects_frozen_python_drift_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["python"] = "/other/python"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=r"execution\.python differs"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    "drift",
    [
        "label",
        "selection_policy",
        "test_policy",
        "source_config",
        "search_parameters",
        "suggest_bounds",
        "max_rounds",
        "max_runs_total",
    ],
)
def test_adaptive_source_rejects_frozen_scientific_contract_before_suggestion_write(
    tmp_path: Path, monkeypatch, drift: str
):
    recipe = _agent_recipe(tmp_path) if drift == "suggest_bounds" else _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    if drift == "label":
        payload.setdefault("inputs", {})["label_name"] = "stage5"
        payload["decisions"]["label_name"]["value"] = "stage5"
    elif drift == "selection_policy":
        payload["evaluation_policy"]["selection_split"] = "train"
        payload["decisions"]["train_val_test_policy"]["value"] = "train"
    elif drift == "test_policy":
        payload["evaluation_policy"]["require_manual_unlock_for_final_test"] = False
    elif drift == "source_config":
        config_path = Path(yaml.safe_load(Path(payload["base_recipe"]).read_text())["inputs"]["config"])
        config = yaml.safe_load(config_path.read_text())
        changed_index = tmp_path / "changed-index.csv"
        changed_index.write_text((tmp_path / "index.csv").read_text())
        config["data"]["finetune_data_index"] = str(changed_index)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    elif drift == "search_parameters":
        payload["search"]["parameters"]["runtime.lr"] = [2e-6]
    elif drift == "suggest_bounds":
        payload["adaptive"]["suggest"]["bounds"]["runtime.lr"] = [6e-7, 2e-6]
    else:
        payload["adaptive"][drift] += 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda *_args: pytest.fail("digest must not run"))

    with pytest.raises(ValueError, match="frozen round 000"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()


@pytest.mark.parametrize(
    ("field", "value", "additional"),
    [
        ("target", "ssh", {"host": "new-host"}),
        ("host", "new-host", {}),
        ("workdir", "/different/runtime", {}),
        ("conda_env", "different-env", {}),
    ],
)
def test_adaptive_source_rejects_frozen_route_drift_before_suggestion_write(
    tmp_path: Path,
    field: str,
    value: str,
    additional: dict[str, str],
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(additional)
    payload["execution"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=rf"execution\.{field} differs from the frozen workflow route"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


def test_adaptive_source_rejects_direct_to_slurm_route_drift_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    manifests.write_rows(workflow_dir / "adaptive" / "digests" / "round_000.csv", [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = _slurm_scheduler()
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=r"execution\.scheduler\.type differs from the frozen workflow route"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()


def test_adaptive_source_rejects_slurm_to_direct_route_drift_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update({"scheduler": _slurm_scheduler(), "gpus_per_run": 1})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    manifests.write_rows(workflow_dir / "adaptive" / "digests" / "round_000.csv", [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = {"type": "direct"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=r"execution\.scheduler\.type differs from the frozen workflow route"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("direct_controller", True), ("partition", "other-gpu"), ("nodelist", "gpu[2]")],
)
def test_adaptive_source_rejects_slurm_routing_drift_before_suggestion_write(tmp_path: Path, field: str, value: object):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update({"scheduler": _slurm_scheduler(), "gpus_per_run": 1})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    manifests.write_rows(workflow_dir / "adaptive" / "digests" / "round_000.csv", [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(
        ValueError,
        match=rf"execution\.scheduler\.{field} differs from the frozen workflow route",
    ):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()


def test_adaptive_source_allows_slurm_resource_and_capacity_updates(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update({"scheduler": _slurm_scheduler(), "gpus_per_run": 1})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    manifests.write_rows(workflow_dir / "adaptive" / "digests" / "round_000.csv", [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"].update(
        {"cpus_per_task": 12, "memory": "96G", "walltime": "02:00:00", "nice": 100}
    )
    payload["execution"]["gpus_per_run"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)
    suggested_execution = yaml.safe_load(suggestion.read_text())["execution"]

    assert suggested_execution["scheduler"]["cpus_per_task"] == 12
    assert suggested_execution["scheduler"]["memory"] == "96G"
    assert suggested_execution["scheduler"]["walltime"] == "02:00:00"
    assert suggested_execution["scheduler"]["nice"] == 100
    assert suggested_execution["gpus_per_run"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [("objective_metric", "val_loss"), ("objective_mode", "min")],
)
def test_adaptive_source_rejects_frozen_objective_drift_before_suggestion_write(tmp_path: Path, field: str, value: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73, "val_loss": 0.42}])
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=rf"adaptive\.{field} differs"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("field", "value"),
    [("objective_metric", "val_loss"), ("objective_mode", "min")],
)
def test_agent_proposal_source_rejects_frozen_objective_drift_before_digest_write(
    tmp_path: Path, field: str, value: str
):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=rf"adaptive\.{field} differs"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert not (workflow_dir / "adaptive" / "digests").exists()
    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()
    assert events_path.read_bytes() == events_before


def test_adaptive_source_accepts_equivalent_default_objective_after_init(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["objective_metric"] = None
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].pop("objective_metric")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)

    workflow = json.loads((workflow_dir / "adaptive" / "workflow.json").read_text())
    assert workflow["objective_metric"] == "test_auroc"
    assert suggestion.exists()


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


def test_adaptive_events_use_frozen_workspace_after_source_root_changes(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    redirected_root = tmp_path / "redirected-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(redirected_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    adaptive_hparam._append_event(workflow_dir, "ownership_probe", {})

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "ownership_probe"
    assert not (redirected_root / "events.jsonl").exists()


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
